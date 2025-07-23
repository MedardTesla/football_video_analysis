from utils import read_video, save_video
from trackers import Tracker

def main():


    video_frame = read_video('input_video/08fd33_4.mp4')

    tracker = Tracker('models/best_telecharger.pt')
    tracks = tracker.get_object_trackers(video_frame,
                                              read_from_stub=False,
                                              stub_path='stubs/track_stubs.pkl')

    # save cropped image of player

    # for track_id, player in tracks['players'][0].items():
    #     bbox = player
    #     frame = video_frame[0]

    #     # crop bbox from frame

    #     cropped_image = frame[]
    
    # draw output
    output_video_frames = tracker.draw_annotations(video_frame, tracks)

    # save video
    save_video(output_video_frames, 'output_video/output_video.avi')


if __name__ == '__main__':
    main()

