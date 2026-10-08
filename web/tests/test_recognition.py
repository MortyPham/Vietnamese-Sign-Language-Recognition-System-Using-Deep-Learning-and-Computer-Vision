"""Behavioral regression tests for inference contracts and sign boundaries."""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from recognition import LiveSession, SequenceCapture
from predict_realtime import mp_frame_to_126
from test import app, decode_image, registry


def hand_vector(offset=0):
    vector = np.zeros((2, 21, 3), dtype=np.float32)
    vector[0, :, 0] = 0.3 + offset
    vector[0, :, 1] = 0.4
    return vector.flatten()


class CaptureTests(unittest.TestCase):
    def test_six_seconds_of_observed_stillness_finishes_sign(self):
        capture = SequenceCapture()
        self.assertEqual(capture.observe(None, 0)["phase"], "waiting_hands")
        capture.observe(hand_vector(), 0.1)
        for timestamp in np.arange(0.3, 6.2, 0.2):
            self.assertNotEqual(capture.observe(hand_vector(), float(timestamp))["phase"], "complete")
        self.assertEqual(capture.observe(hand_vector(), 6.4)["phase"], "complete")

    def test_moving_sign_is_not_cut_at_a_fixed_interval(self):
        capture = SequenceCapture()
        for index in range(100):
            state = capture.observe(hand_vector(0.08 * (index % 2)), index / 6)
            self.assertEqual(state["phase"], "signing")
        self.assertFalse(capture.done)

    def test_motion_resets_the_pause(self):
        capture = SequenceCapture()
        capture.observe(hand_vector(), 0)
        for timestamp in np.arange(0.2, 4.1, 0.2):
            capture.observe(hand_vector(), float(timestamp))
        self.assertEqual(capture.observe(hand_vector(0.1), 4.2)["phase"], "signing")
        self.assertEqual(capture.observe(hand_vector(0.1), 4.4)["remaining"], 6)

    def test_tracking_loss_and_network_gaps_cannot_finish_sign(self):
        capture = SequenceCapture()
        capture.observe(hand_vector(), 0)
        capture.observe(hand_vector(), 0.2)
        for timestamp in np.arange(0.4, 3.1, 0.2):
            capture.observe(hand_vector(), float(timestamp))
        self.assertEqual(capture.observe(None, 3.2)["phase"], "waiting_hands")
        self.assertEqual(capture.observe(hand_vector(), 3.4)["phase"], "signing")
        self.assertEqual(capture.observe(hand_vector(), 3.6)["remaining"], 6)
        self.assertEqual(capture.observe(hand_vector(), 12)["phase"], "signing")
        self.assertFalse(capture.done)

    def test_small_jitter_does_not_restart_pause(self):
        capture = SequenceCapture()
        capture.observe(hand_vector(), 0)
        for index in range(1, 40):
            state = capture.observe(hand_vector(0.003 * (index % 2)), index / 6)
        self.assertEqual(state["phase"], "complete")

    def test_gradual_motion_is_compared_with_anchor(self):
        capture = SequenceCapture()
        capture.observe(hand_vector(), 0)
        phases = [capture.observe(hand_vector(index * 0.005), index / 6)["phase"] for index in range(1, 20)]
        self.assertGreater(phases.count("signing"), 2)

    def test_holding_tail_is_excluded_from_phrase_evidence(self):
        capture = SequenceCapture()
        for index in range(12):
            capture.observe(hand_vector(0.08 * (index % 2)), index / 6)
        for timestamp in np.arange(2.1, 8.5, 0.2):
            capture.observe(hand_vector(0.08), float(timestamp))
        self.assertLess(len(capture.active_vectors()), 16)
        self.assertTrue(capture.done)

    def test_capture_safety_limit_returns_timeout(self):
        capture = SequenceCapture()
        capture.observe(hand_vector(), 0)
        self.assertEqual(capture.observe(hand_vector(0.1), 46)["phase"], "timeout")

    def test_unmirrored_handedness_matches_dataset_order(self):
        points = [SimpleNamespace(x=.2, y=.4, z=.01) for _ in range(21)]
        results = SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=points)],
            multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label="Right")])])
        raw = mp_frame_to_126(results)
        selfie = mp_frame_to_126(results, mirrored=True)
        self.assertTrue(np.any(raw[:63]))
        self.assertFalse(np.any(raw[63:]))
        self.assertFalse(np.any(selfie[:63]))
        self.assertTrue(np.any(selfie[63:]))

    def test_live_session_emits_result_only_after_pause(self):
        tracker = Mock()
        tracker.process.return_value = SimpleNamespace(multi_hand_landmarks=[object()])
        model = Mock()
        model.image_result.return_value = {"status": "recognized", "label": "A"}
        with patch("recognition.hand_tracker", return_value=tracker), patch("recognition.mp_frame_to_126", return_value=hand_vector()):
            session = LiveSession("letters")
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            for timestamp in np.arange(0, 6.2, 0.2):
                with patch("recognition.time.monotonic", return_value=float(timestamp)):
                    result = session.process(frame, model)
                self.assertNotIn("result", result)
                model.image_result.assert_not_called()
            with patch("recognition.time.monotonic", return_value=6.5):
                result = session.process(frame, model)
            self.assertEqual(result["result"]["label"], "A")
            model.image_result.assert_called_once()
            session.close()


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_loaded_models_and_real_dataset_images(self):
        modes = {item["id"]: item for item in self.client.get("/api/models").json()["models"]}
        self.assertEqual(modes["letters"]["count"], 22)
        self.assertTrue(modes["sentences"]["available"])
        for name, expected in [("sample-1.jpg", "Y"), ("sample-2.jpg", "X"), ("sample-3.jpg", "D")]:
            content = (Path(__file__).resolve().parents[1] / "assets" / "samples" / name).read_bytes()
            response = self.client.post("/api/upload", data={"mode": "letters"}, files={"file": (name, content, "image/jpeg")})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["label"], expected)

    def test_sentence_photo_has_actionable_error(self):
        response = self.client.post("/api/upload", data={"mode": "sentences"}, files={"file": ("photo.jpg", b"data", "image/jpeg")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("video", response.json()["detail"])

    def test_demonstrations_match_supported_sentence_labels(self):
        from urllib.parse import urlparse
        response = self.client.get("/assets/signs/sentence-videos.json")
        self.assertEqual(response.status_code, 200)
        videos = response.json()["demonstrations"]
        sentence_model = next(model for model in self.client.get("/api/models").json()["models"] if model["id"] == "sentences")
        labels = [video["label"] for video in videos]
        self.assertGreater(len(labels), 0)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(set(labels).issubset(sentence_model["labels"]))
        for video in videos:
            self.assertEqual(video["provider"], "youtube")
            self.assertRegex(video["video_id"], r"^[A-Za-z0-9_-]{11}$")
            self.assertEqual(urlparse(video["source_url"]).scheme, "https")
            self.assertIn(urlparse(video["source_url"]).hostname, {"tokyolife.vn", "talkingdictionary.swarthmore.edu"})
            self.assertEqual(video["watch_url"], "https://www.youtube.com/watch?v=" + video["video_id"])

    def test_corrupt_image_rejected(self):
        response = self.client.post("/api/upload", data={"mode": "letters"}, files={"file": ("photo.jpg", b"not an image", "image/jpeg")})
        self.assertEqual(response.status_code, 400)

    def test_empty_upload_rejected(self):
        response = self.client.post("/api/upload", files={"file": ("photo.png", b"", "image/png")})
        self.assertEqual(response.status_code, 400)

    def test_blank_image_does_not_invent_translation(self):
        content = io.BytesIO()
        Image.new("RGB", (200, 200), "black").save(content, format="PNG")
        response = self.client.post("/api/upload", files={"file": ("blank.png", content.getvalue(), "image/png")})
        self.assertEqual(response.json()["status"], "no_sign")
        self.assertIsNone(response.json()["label"])

    def test_actual_video_decode_with_no_hands(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blank.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (128, 128))
            self.assertTrue(writer.isOpened())
            for _ in range(24):
                writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
            writer.release()
            response = self.client.post("/api/upload", data={"mode": "sentences"}, files={"file": ("blank.mp4", path.read_bytes(), "video/mp4")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "no_sign")

    def test_session_lifecycle_and_blank_frame(self):
        response = self.client.post("/api/live/session", data={"mode": "letters"})
        self.assertEqual(response.status_code, 200)
        key = response.json()["id"]
        content = io.BytesIO()
        Image.new("RGB", (128, 128), "black").save(content, format="JPEG")
        response = self.client.post(f"/api/live/{key}/frame", files={"file": ("frame.jpg", content.getvalue(), "image/jpeg")})
        self.assertEqual(response.json()["phase"], "waiting_hands")
        self.assertNotIn("result", response.json())
        self.assertEqual(self.client.delete(f"/api/live/{key}").status_code, 200)
        self.assertEqual(self.client.post(f"/api/live/{key}/frame", files={"file": ("frame.jpg", content.getvalue(), "image/jpeg")}).status_code, 404)

    def test_real_sample_tracking_finishes_and_recognizes_held_sign(self):
        content = (Path(__file__).resolve().parents[1] / "assets" / "samples" / "sample-1.jpg").read_bytes()
        session = LiveSession("letters")
        try:
            result = None
            for timestamp in np.arange(0, 12, 1 / 6):
                with patch("recognition.time.monotonic", return_value=float(timestamp)):
                    result = session.process(decode_image(content), registry)
                if result["phase"] == "complete":
                    break
                self.assertNotIn("result", result)
            self.assertEqual(result["phase"], "complete", result)
            self.assertEqual(result["result"]["label"], "Y")
        finally:
            session.close()

    def test_navigation_opens_distinct_home_upload_and_webcam_pages(self):
        for path, page in [("/", "home"), ("/home.html", "home"), ("/upload.html", "upload"), ("/webcam.html", "webcam")]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn("VSL Studio", response.text)
            self.assertNotIn("iframe", response.text)
            self.assertIn(f'data-page="{page}"', response.text)
            if page == "home":
                self.assertNotIn('id="translate-button"', response.text)
            else:
                self.assertNotIn('id="about"', response.text)


if __name__ == "__main__":
    unittest.main()
