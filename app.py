from pathlib import Path
import pickle
import tempfile

import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "rf_best_model.pkl"


def extract_features(y, sr=22050):
    """Keep feature names, order, and sample rate consistent with training."""
    import librosa

    features = {
        "RMSE": float(librosa.feature.rms(y=y).mean()),
        "SPECTRAL_CENTROID": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
        "SPECTRAL_BANDWIDTH": float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()),
        "ROLLOFF": float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean()),
        "ZERO_CROSSING_RATE": float(librosa.feature.zero_crossing_rate(y=y).mean()),
    }
    for index, values in enumerate(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)):
        features[f"MFCC_FEATURE_{index}"] = float(values.mean())
    return features


def process_video(input_path, work_dir, model, top_db, threshold):
    # Defer the audio/video stack until processing so the upload screen can load
    # without importing librosa/Numba and MoviePy on a cold server.
    import librosa
    from moviepy import VideoFileClip
    from PIL import Image, ImageDraw, ImageFont

    audio_path = work_dir / "audio.wav"
    output_path = work_dir / "output.mp4"
    with VideoFileClip(str(input_path)) as video:
        if video.audio is None:
            raise ValueError("The uploaded video has no audio track.")
        video.audio.write_audiofile(str(audio_path), codec="pcm_s16le", logger=None)
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        if y.size == 0:
            raise ValueError("The audio track is empty.")
        # librosa's relative silence threshold alone can classify all-zero audio
        # as non-silent, so explicitly handle it first.
        intervals = (
            librosa.effects.split(y, top_db=top_db)
            if np.max(np.abs(y)) > 1e-8
            else np.empty((0, 2), dtype=int)
        )
        rows = []
        for start, end in intervals:
            start_sec = float(start / sr)
            end_sec = min(float(end / sr), float(video.duration))
            if end_sec <= start_sec:
                continue
            segment = librosa.resample(y[start:end], orig_sr=sr, target_sr=22050)
            frame = pd.DataFrame([extract_features(segment)])
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
                missing = set(expected) - set(frame.columns)
                if missing:
                    raise ValueError(f"Model expects unavailable features: {sorted(missing)}")
                frame = frame.loc[:, expected]
            else:
                frame = frame.to_numpy()
            probabilities = np.asarray(model.predict_proba(frame))
            if probabilities.ndim != 2 or probabilities.shape[0] != 1:
                raise ValueError("Expected a single-output classification model.")
            best = int(np.argmax(probabilities[0]))
            confidence = float(probabilities[0, best])
            label = str(model.classes_[best]) if confidence > threshold else "Unknown"
            rows.append({"start_time": start_sec, "end_time": end_sec,
                         "prediction": label, "confidence": confidence})

        results = pd.DataFrame(rows, columns=["start_time", "end_time", "prediction", "confidence"])
        starts = results["start_time"].to_numpy(dtype=float)
        font = ImageFont.load_default(size=max(12, min(24, int(video.w / 35))))

        def overlay(get_frame, t):
            image = get_frame(t).copy()
            index = int(np.searchsorted(starts, t, side="right")) - 1
            if index >= 0 and t < rows[index]["end_time"]:
                row = rows[index]
                text = f"Prediction: {row['prediction']} ({row['confidence']:.0%})"
                canvas = Image.fromarray(image)
                ImageDraw.Draw(canvas).text(
                    (10, 10), text, font=font, fill=(0, 255, 0),
                    stroke_width=1, stroke_fill=(0, 0, 0),
                )
                image = np.asarray(canvas)
            return image

        annotated = video.transform(overlay)
        try:
            annotated.write_videofile(
                str(output_path), codec="libx264", audio_codec="aac",
                threads=2,
                temp_audiofile=str(work_dir / "encoded_audio.m4a"),
                ffmpeg_params=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart"],
                logger=None,
            )
        finally:
            annotated.close()
    return output_path.read_bytes(), results


def main():
    st.title("Kulintang Music Transcription")
    st.write("Upload a video to classify non-silent intervals and overlay predictions.")
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov"])
    top_db = st.slider("Silence threshold (dB below peak)", 5, 60, 20)
    threshold = st.slider("Minimum prediction confidence", 0.0, 1.0, 0.7, 0.05)
    if st.button("Process video", disabled=uploaded is None):
        st.session_state.pop("processed_result", None)
        if not MODEL_PATH.is_file():
            st.error(f"Place rf_best_model.pkl next to app.py: {MODEL_PATH}")
            return
        try:
            # Only load a model you trust: pickle files can execute Python code.
            with MODEL_PATH.open("rb") as handle:
                model = pickle.load(handle)
            if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
                raise ValueError("The model must support predict_proba and classes_.")
            with st.spinner("Processing video..."):
                with tempfile.TemporaryDirectory() as directory:
                    work_dir = Path(directory)
                    input_path = work_dir / ("input" + Path(uploaded.name).suffix.lower())
                    input_path.write_bytes(uploaded.getvalue())
                    video_bytes, results = process_video(input_path, work_dir, model, top_db, threshold)
                st.session_state["processed_result"] = (video_bytes, results)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            return

    if "processed_result" in st.session_state:
        video_bytes, results = st.session_state["processed_result"]
        st.success("Video processing complete.")
        if results.empty:
            st.info("No non-silent intervals were detected.")
        st.video(video_bytes, format="video/mp4")
        st.dataframe(results, hide_index=True)
        st.download_button("Download processed video", video_bytes,
                           "output_with_detections.mp4", "video/mp4", on_click="ignore")
        st.download_button("Download predictions", results.to_csv(index=False).encode("utf-8"),
                           "predictions.csv", "text/csv", on_click="ignore")


if __name__ == "__main__":
    main()
