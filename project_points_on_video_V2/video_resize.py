import moviepy.editor as mp


if __name__ == '__main__':
    video_path = r"C:\Users\Ofir Gilad\PycharmProjects\YOLOv8\dataloop_annotations\girls.mp4"
    clip = mp.VideoFileClip(video_path)
    clip_resized = clip.resize(width=1024, height=1280)
    clip_resized.write_videofile("output_girls.mp4")
