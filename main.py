import sys
import pandas as pd
from video_extract_csv import video_csv
from classify_predict import config_model, predict
from correction_predict import corr_predict


def main():
    if len(sys.argv) != 2:
        print("Your Input script should be : python your_script_name.py <video_file> :")
        sys.exit(1)

    video_file = sys.argv[1]
    print(f"Video file : {video_file}")
    video_csv(video_file)
    print()
    print('The csv file has been made and keypoints have been extracted')
    print()

    model = config_model()

    data = pd.read_csv('combined_new.csv')

    pose = predict(data, model)

    pose_name = {0:'cobra', 1:'tree', 2:'goddess', 3:'chair', 4:'downdog', 5:'warrior'}

    pose = pose_name[pose]

    print()
    print('The correct pose predicted is : ',pose)
    print()

        

    corr_predict(pose, data)


if __name__ == "__main__":
    main()



    