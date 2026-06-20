import pytest

from oasis_forge.streaming import (
	ACTION_MAGIC,
	DTYPE_F16_LE,
	LATENT_MAGIC,
	ActionPacket,
	LatentPacket,
	expected_latent_payload_bytes,
	local_action_from_json,
	pack_action,
	pack_latent,
	unpack_action,
	unpack_latent,
)


def test_action_packet_roundtrip():
	packet = ActionPacket(
		player_id=7,
		seq=42,
		client_time_ns=123_456,
		buttons=0b101,
		mouse_dx=1.25,
		mouse_dy=-0.5,
	)
	decoded = unpack_action(pack_action(packet))
	assert decoded.player_id == packet.player_id
	assert decoded.seq == packet.seq
	assert decoded.client_time_ns == packet.client_time_ns
	assert decoded.buttons == packet.buttons
	assert decoded.mouse_dx == pytest.approx(packet.mouse_dx)
	assert decoded.mouse_dy == pytest.approx(packet.mouse_dy)


def test_latent_packet_roundtrip():
	payload = bytes(expected_latent_payload_bytes(2, 3, 16))
	packet = LatentPacket(
		player_id=1,
		seq=9,
		server_time_ns=987_654,
		latent_height=2,
		latent_width=3,
		channels=16,
		dtype=DTYPE_F16_LE,
		payload=payload,
	)
	decoded = unpack_latent(pack_latent(packet))
	assert decoded.player_id == packet.player_id
	assert decoded.seq == packet.seq
	assert decoded.server_time_ns == packet.server_time_ns
	assert decoded.latent_height == packet.latent_height
	assert decoded.latent_width == packet.latent_width
	assert decoded.channels == packet.channels
	assert decoded.dtype == packet.dtype
	assert decoded.payload == payload


def test_bad_magic_rejected():
	action_data = bytearray(pack_action(ActionPacket(player_id=1, seq=1, client_time_ns=1)))
	action_data[:4] = b"NOPE"
	with pytest.raises(ValueError, match="magic"):
		unpack_action(bytes(action_data))

	latent_payload = bytes(expected_latent_payload_bytes(1, 1, 16))
	latent_data = bytearray(
		pack_latent(
			LatentPacket(
				player_id=1,
				seq=1,
				server_time_ns=1,
				latent_height=1,
				latent_width=1,
				channels=16,
				dtype=DTYPE_F16_LE,
				payload=latent_payload,
			)
		)
	)
	latent_data[:4] = b"NOPE"
	with pytest.raises(ValueError, match="magic"):
		unpack_latent(bytes(latent_data))


def test_local_action_json_to_packet():
	packet = local_action_from_json(
		b'{"buttons": 3, "mouse_dx": 0.25, "mouse_dy": -0.75}',
		player_id=4,
		seq=11,
	)
	assert packet.player_id == 4
	assert packet.seq == 11
	assert packet.buttons == 3
	assert packet.mouse_dx == pytest.approx(0.25)
	assert packet.mouse_dy == pytest.approx(-0.75)


def test_protocol_magics_are_distinct():
	assert ACTION_MAGIC != LATENT_MAGIC

