import numpy as np
import cupy as cp

from cv2 import VideoWriter, cvtColor, COLOR_RGB2BGR  # NOQA
from moviepy import VideoFileClip
from cupyx.scipy import ndimage as ndi
from decord import VideoReader
from decord import cpu
from tqdm import tqdm


key_buffer: cp.ndarray


def build_frame(frame: cp.ndarray, division, frame_size) -> cp.ndarray:  # NOQA
    height_step = frame.shape[0] // division
    width_step = frame.shape[1] // division

    frame = frame.reshape(division, height_step, division, width_step, 3)
    frame = frame.transpose(0, 2, 1, 3, 4)
    frame = frame.reshape(-1, height_step, width_step, 3)

    frame = cp.broadcast_to(
        cp.expand_dims(frame, axis=1),
        (frame.shape[0], frame_size, height_step, width_step, 3)
    )

    frame = cp.maximum(frame, key_buffer) - cp.minimum(frame, key_buffer)
    index = cp.argmin(cp.sum(frame, axis=(2, 3, 4)), axis=1)
    frame = key_buffer[index].astype(cp.uint8)

    frame = frame.reshape(division, division, height_step, width_step, 3)
    frame = frame.transpose(0, 2, 1, 3, 4)
    frame = frame.reshape(height_step * division, width_step * division, 3)

    return frame


def bad_resolution(h, w, d):
    return h % d or w % d


if __name__ == '__main__':
    # VARIABLES =============================================
    input_file = 'input.mp4'
    output_file = 'output.mp4'

    # VRAM 거지ㅜㅜ
    division = 20
    max_buffer_size = 500
    batch_size = 32
    # =======================================================

    print(f"Reading video... ({input_file})")
    video_reader = VideoReader(input_file, ctx=cpu(0))
    frames = len(video_reader)
    max_buffer_size = min(max_buffer_size, frames)

    if bad_resolution(*video_reader[0].shape[:2], division):
        print("Bad Resolution!")
        exit(1)

    indices = np.linspace(0, frames - 1, num=max_buffer_size, dtype=int)
    buffer = video_reader.get_batch(indices)  # NOQA
    buffer = cp.asarray(buffer.asnumpy(), dtype=cp.uint8)

    zoom_factors = (1, 1 / division, 1 / division, 1)
    key_buffer = ndi.zoom(buffer, zoom_factors, order=1)
    del buffer

    print(f"Buffer Shape: {key_buffer.shape}")

    fourcc = VideoWriter.fourcc(*'mp4v')
    writer = VideoWriter(
        output_file,
        fourcc,
        video_reader.get_avg_fps(),
        video_reader[0].shape[:2][::-1]
    )

    for i in tqdm(range(0, frames, batch_size)):
        batch_index = list(range(i, min(i + batch_size, frames)))
        if not batch_index:
            break

        batch = video_reader.get_batch(batch_index).asnumpy()
        batch = cp.asarray(batch, dtype=cp.uint8)

        for j in range(batch.shape[0]):
            f = build_frame(batch[j], division, max_buffer_size)
            f = f.get()
            f = cvtColor(f, COLOR_RGB2BGR)

            writer.write(f)

    writer.release()

    video_clip = VideoFileClip(output_file)
    audio_clip = VideoFileClip(input_file).audio

    video_clip.audio = audio_clip
    video_clip.write_videofile(f"audio_{output_file}")

    print(f"Video saved! ({output_file})")
