import subprocess

import numpy as np
import cupy as cp

from cv2 import VideoWriter, cvtColor, COLOR_RGB2BGR  # NOQA
from imageio_ffmpeg import get_ffmpeg_exe
from cupyx.scipy import ndimage as ndi
from decord import VideoReader
from decord import cpu
from tqdm import tqdm


key_buffer: cp.ndarray
key_buffer_flat: cp.ndarray
key_buffer_squared: cp.ndarray


def build_frame(
        frame: cp.ndarray,
        divisor: int,
        previous_index: cp.ndarray | None,
        temporal_threshold: float
) -> tuple[cp.ndarray, cp.ndarray]:  # NOQA
    height_step = frame.shape[0] // divisor
    width_step = frame.shape[1] // divisor

    frame = frame.reshape(divisor, height_step, divisor, width_step, 3)
    frame = frame.transpose(0, 2, 1, 3, 4)
    frame = frame.reshape(-1, height_step * width_step * 3).astype(cp.float32)

    frame_squared = cp.sum(frame * frame, axis=1)

    distance = frame_squared[:, None] + key_buffer_squared[None, :] - (2.0 * frame @ key_buffer_flat.T)
    distance = cp.maximum(distance, 0.0)

    index = cp.argmin(distance, axis=1)
    if previous_index is not None and temporal_threshold > 0:
        tile_index = cp.arange(index.shape[0])
        best_distance = distance[tile_index, index]
        previous_distance = distance[tile_index, previous_index]

        index = cp.where(
            best_distance < previous_distance * (1.0 - temporal_threshold),
            index,
            previous_index
        )

    frame = key_buffer[index]
    frame = frame.reshape(divisor, divisor, height_step, width_step, 3)
    frame = frame.transpose(0, 2, 1, 3, 4)
    frame = frame.reshape(height_step * divisor, width_step * divisor, 3)

    return frame, index


def is_bad_resolution(height: int, width: int, divisor: int) -> bool:
    return height % divisor != 0 or width % divisor != 0


def main(
        input_file: str,
        source_file: str,
        output_file: str,
        divisor: int,
        max_buffer_size: int,
        batch_size: int,
        temporal_threshold: float
):
    global key_buffer, key_buffer_flat, key_buffer_squared

    print(f"Reading videos... ({input_file}, {source_file})")
    input_reader = VideoReader(input_file, ctx=cpu(0))
    source_reader = VideoReader(source_file, ctx=cpu(0))

    input_resolution = input_reader[0].shape[:2]
    source_resolution = source_reader[0].shape[:2]

    if input_resolution != source_resolution:
        print(f"Resolution Error: Input = {input_resolution[::-1]}, Source = {source_resolution[::-1]}")
        exit(1)

    if is_bad_resolution(*input_resolution, divisor):
        print("Divisor Error")
        exit(1)

    frames = len(input_reader)
    source_frames = len(source_reader)
    max_buffer_size = min(max_buffer_size, source_frames)

    indices = np.linspace(0, source_frames - 1, num=max_buffer_size, dtype=int)
    buffer = source_reader.get_batch(indices)  # NOQA
    buffer = cp.asarray(buffer.asnumpy())

    zoom_factors = (1, 1 / divisor, 1 / divisor, 1)
    key_buffer = ndi.zoom(buffer, zoom_factors, order=1).astype(cp.uint8)
    key_buffer_flat = key_buffer.reshape(max_buffer_size, -1).astype(cp.float32)
    key_buffer_squared = cp.sum(key_buffer_flat * key_buffer_flat, axis=1)

    del buffer, indices, zoom_factors

    print(f"Buffer Shape: {key_buffer.shape}")
    fourcc = VideoWriter.fourcc(*'mp4v')
    writer = VideoWriter(
        output_file,
        fourcc,
        input_reader.get_avg_fps(),
        input_resolution[::-1]
    )

    previous_index: cp.ndarray | None = None

    for i in tqdm(range(0, frames, batch_size)):
        batch_index = list(range(i, min(i + batch_size, frames)))
        if not batch_index:
            break

        batch = input_reader.get_batch(batch_index).asnumpy()
        batch = cp.asarray(batch, dtype=cp.uint8)

        for j in range(batch.shape[0]):
            frame, previous_index = build_frame(
                batch[j],
                divisor,
                previous_index,
                temporal_threshold
            )
            frame = frame.get()
            frame = cvtColor(frame, COLOR_RGB2BGR)

            writer.write(frame)

    writer.release()

    del key_buffer, key_buffer_flat, key_buffer_squared, input_reader, source_reader

    audio_output_file = f"audio_{output_file}"
    subprocess.run(
        [
            get_ffmpeg_exe(),
            '-y',
            '-i', output_file,
            '-i', input_file,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            audio_output_file
        ],
        check=True
    )

    print(f"Video saved! ({audio_output_file})")


if __name__ == "__main__":
    main(
        input_file='input.mp4',
        source_file='source.mp4',
        output_file='output.mp4',
        divisor=20,
        max_buffer_size=512,
        batch_size=32,
        temporal_threshold=0.03
    )
