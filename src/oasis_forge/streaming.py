"""Split-serving wire protocol and lightweight relay loops.

The runtime split is:

	server/B300: receive player actions, generate next latent chunks
	laptop: receive latent chunks, forward raw f16 latents to the local Metal decoder

The Metal decoder intentionally keeps its UDP input simple: one raw latent datagram. This module
wraps network traffic between server and laptop with small headers, then the laptop bridge strips
the header before forwarding raw bytes to the decoder.
"""

from __future__ import annotations

import json
import select
import socket
import struct
import time
from dataclasses import dataclass
from typing import Protocol


ACTION_MAGIC = b"KVCA"
LATENT_MAGIC = b"KVCL"
PROTOCOL_VERSION = 1
DTYPE_F16_LE = 1
UDP_SAFE_BYTES = 60_000

ACTION_STRUCT = struct.Struct("<4sBBHQQIff")
LATENT_STRUCT = struct.Struct("<4sBBHQQHHHHI")


@dataclass(frozen=True)
class ActionPacket:
	player_id: int
	seq: int
	client_time_ns: int
	buttons: int = 0
	mouse_dx: float = 0.0
	mouse_dy: float = 0.0
	flags: int = 0


@dataclass(frozen=True)
class LatentPacket:
	player_id: int
	seq: int
	server_time_ns: int
	latent_height: int
	latent_width: int
	channels: int
	dtype: int
	payload: bytes
	flags: int = 0


def monotonic_ns() -> int:
	return time.monotonic_ns()


def expected_latent_payload_bytes(latent_height: int, latent_width: int, channels: int = 16) -> int:
	return latent_height * latent_width * channels * 2


def pack_action(packet: ActionPacket) -> bytes:
	return ACTION_STRUCT.pack(
		ACTION_MAGIC,
		PROTOCOL_VERSION,
		packet.flags,
		packet.player_id,
		packet.seq,
		packet.client_time_ns,
		packet.buttons,
		packet.mouse_dx,
		packet.mouse_dy,
	)


def unpack_action(data: bytes) -> ActionPacket:
	if len(data) != ACTION_STRUCT.size:
		raise ValueError(f"action packet is {len(data)} bytes; expected {ACTION_STRUCT.size}")
	magic, version, flags, player_id, seq, client_time_ns, buttons, mouse_dx, mouse_dy = (
		ACTION_STRUCT.unpack(data)
	)
	if magic != ACTION_MAGIC:
		raise ValueError("bad action packet magic")
	if version != PROTOCOL_VERSION:
		raise ValueError(f"unsupported action protocol version {version}")
	return ActionPacket(player_id, seq, client_time_ns, buttons, mouse_dx, mouse_dy, flags)


def pack_latent(packet: LatentPacket) -> bytes:
	header = LATENT_STRUCT.pack(
		LATENT_MAGIC,
		PROTOCOL_VERSION,
		packet.flags,
		packet.player_id,
		packet.seq,
		packet.server_time_ns,
		packet.latent_height,
		packet.latent_width,
		packet.channels,
		packet.dtype,
		len(packet.payload),
	)
	return header + packet.payload


def unpack_latent(data: bytes) -> LatentPacket:
	if len(data) < LATENT_STRUCT.size:
		raise ValueError("latent packet is shorter than header")
	header = data[: LATENT_STRUCT.size]
	payload = data[LATENT_STRUCT.size :]
	(
		magic,
		version,
		flags,
		player_id,
		seq,
		server_time_ns,
		latent_height,
		latent_width,
		channels,
		dtype,
		payload_len,
	) = LATENT_STRUCT.unpack(header)
	if magic != LATENT_MAGIC:
		raise ValueError("bad latent packet magic")
	if version != PROTOCOL_VERSION:
		raise ValueError(f"unsupported latent protocol version {version}")
	if payload_len != len(payload):
		raise ValueError(f"latent payload is {len(payload)} bytes; expected {payload_len}")
	expected = expected_latent_payload_bytes(latent_height, latent_width, channels)
	if dtype == DTYPE_F16_LE and payload_len != expected:
		raise ValueError(f"f16 latent payload is {payload_len} bytes; expected {expected}")
	return LatentPacket(
		player_id,
		seq,
		server_time_ns,
		latent_height,
		latent_width,
		channels,
		dtype,
		payload,
		flags,
	)


def local_action_from_json(data: bytes, player_id: int, seq: int) -> ActionPacket:
	"""Parse a laptop-local action datagram.

	Expected JSON:
		{"buttons": 3, "mouse_dx": 0.5, "mouse_dy": -0.25}
	"""
	obj = json.loads(data.decode("utf-8"))
	return ActionPacket(
		player_id=player_id,
		seq=seq,
		client_time_ns=monotonic_ns(),
		buttons=int(obj.get("buttons", 0)),
		mouse_dx=float(obj.get("mouse_dx", 0.0)),
		mouse_dy=float(obj.get("mouse_dy", 0.0)),
	)


class LatentGenerator(Protocol):
	def push_action(self, action: ActionPacket) -> None:
		"""Accept the latest action for one player."""

	def next_latent(self, player_id: int) -> bytes:
		"""Return one raw NHWTC f16 latent payload for a player."""


class ZeroLatentGenerator:
	"""Test generator used before the real JAX/B300 loop is wired in."""

	def __init__(self, latent_height: int, latent_width: int, channels: int = 16):
		self.payload = bytes(expected_latent_payload_bytes(latent_height, latent_width, channels))
		self.latest_actions: dict[int, ActionPacket] = {}

	def push_action(self, action: ActionPacket) -> None:
		self.latest_actions[action.player_id] = action

	def next_latent(self, player_id: int) -> bytes:
		return self.payload


def run_central_server(
	action_host: str = "0.0.0.0",
	action_port: int = 7780,
	client_latent_port: int = 7781,
	latent_height: int = 28,
	latent_width: int = 50,
	channels: int = 16,
	fps: float = 12.0,
	generator: LatentGenerator | None = None,
) -> None:
	"""Receive actions from laptops and stream generated latents back to them."""
	payload_bytes = expected_latent_payload_bytes(latent_height, latent_width, channels)
	packet_bytes = LATENT_STRUCT.size + payload_bytes
	if packet_bytes > UDP_SAFE_BYTES:
		raise ValueError(
			f"latent packet would be {packet_bytes} bytes; keep UDP packets below "
			f"{UDP_SAFE_BYTES} or add fragmentation/TCP"
		)
	generator = generator or ZeroLatentGenerator(latent_height, latent_width, channels)

	action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	action_sock.bind((action_host, action_port))
	action_sock.setblocking(False)
	latent_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	clients: dict[int, tuple[str, int]] = {}
	seq_by_player: dict[int, int] = {}
	period = 1.0 / fps
	next_send = time.monotonic()
	print(
		f"central server listening actions on {action_host}:{action_port}; "
		f"streaming {latent_height}x{latent_width}x{channels} f16 latents to "
		f"client port {client_latent_port} at {fps:.2f} fps",
		flush=True,
	)
	while True:
		while select.select([action_sock], [], [], 0.0)[0]:
			data, addr = action_sock.recvfrom(2048)
			try:
				action = unpack_action(data)
			except ValueError as exc:
				print(f"dropped bad action packet from {addr}: {exc}", flush=True)
				continue
			dst = (addr[0], client_latent_port)
			if clients.get(action.player_id) != dst:
				print(f"registered player {action.player_id} at {dst[0]}:{dst[1]}", flush=True)
			clients[action.player_id] = dst
			generator.push_action(action)

		now = time.monotonic()
		if now >= next_send:
			for player_id, dst in clients.items():
				seq = seq_by_player.get(player_id, 0)
				payload = generator.next_latent(player_id)
				packet = pack_latent(
					LatentPacket(
						player_id=player_id,
						seq=seq,
						server_time_ns=monotonic_ns(),
						latent_height=latent_height,
						latent_width=latent_width,
						channels=channels,
						dtype=DTYPE_F16_LE,
						payload=payload,
					)
				)
				latent_sock.sendto(packet, dst)
				seq_by_player[player_id] = seq + 1
			next_send = now + period
		time.sleep(0.002)


def run_laptop_bridge(
	server_host: str,
	player_id: int,
	server_action_port: int = 7780,
	listen_latent_port: int = 7781,
	local_action_port: int = 7790,
	decoder_host: str = "127.0.0.1",
	decoder_port: int = 7777,
	action_fps: float = 60.0,
) -> None:
	"""Forward laptop input to the server and server latents to the local Metal decoder."""
	server_addr = (server_host, server_action_port)
	decoder_addr = (decoder_host, decoder_port)
	action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	local_action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	local_action_sock.bind(("127.0.0.1", local_action_port))
	local_action_sock.setblocking(False)
	latent_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	latent_sock.bind(("0.0.0.0", listen_latent_port))
	latent_sock.setblocking(False)
	decoder_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

	current_action = ActionPacket(player_id=player_id, seq=0, client_time_ns=monotonic_ns())
	action_seq = 0
	period = 1.0 / action_fps
	next_action_send = time.monotonic()
	print(
		f"laptop bridge player={player_id}: local actions 127.0.0.1:{local_action_port} -> "
		f"server {server_addr[0]}:{server_addr[1]}; latents :{listen_latent_port} -> "
		f"decoder {decoder_addr[0]}:{decoder_addr[1]}",
		flush=True,
	)
	while True:
		while select.select([local_action_sock], [], [], 0.0)[0]:
			data, _ = local_action_sock.recvfrom(2048)
			try:
				current_action = local_action_from_json(data, player_id, action_seq)
			except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
				print(f"dropped bad local action datagram: {exc}", flush=True)

		now = time.monotonic()
		if now >= next_action_send:
			action = ActionPacket(
				player_id=player_id,
				seq=action_seq,
				client_time_ns=monotonic_ns(),
				buttons=current_action.buttons,
				mouse_dx=current_action.mouse_dx,
				mouse_dy=current_action.mouse_dy,
			)
			action_sock.sendto(pack_action(action), server_addr)
			action_seq += 1
			next_action_send = now + period

		while select.select([latent_sock], [], [], 0.0)[0]:
			data, addr = latent_sock.recvfrom(UDP_SAFE_BYTES + 4096)
			try:
				latent = unpack_latent(data)
			except ValueError as exc:
				print(f"dropped bad latent packet from {addr}: {exc}", flush=True)
				continue
			if latent.player_id != player_id:
				continue
			decoder_sock.sendto(latent.payload, decoder_addr)

		time.sleep(0.001)

