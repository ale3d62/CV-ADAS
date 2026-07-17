import pandas as pd
import io
import os
from PIL import Image
import glob


def extract_front_camera_images(inputFolder, outputBaseDir):
    parquetFiles = glob.glob(os.path.join(inputFolder, "*.parquet"))

    if not parquetFiles:
        print(f"No .parquet files found in {inputFolder}")
        return

    print(f"Found {len(parquetFiles)} parquet camera files. Extracting images...")

    for parquetFiles in parquetFiles:
        filename = os.path.splitext(os.path.basename(parquetFiles))[0]
        outputSubfolder = os.path.join(outputBaseDir, filename)

        if not os.path.exists(outputSubfolder):
            os.makedirs(outputSubfolder)

        try:
            cols = ['[CameraImageComponent].image', 'key.frame_timestamp_micros', 'key.camera_name']
            df = pd.read_parquet(parquetFiles, columns=cols)

            df_front = df[df['key.camera_name'] == 1]

            nImages = len(df_front)
            if nImages > 0:
                print(f"Processing {filename}: extracting {nImages} images...")

                for i, (_, row) in enumerate(df_front.iterrows()):
                    imageBytes = row['[CameraImageComponent].image']
                    ts = str(row['key.frame_timestamp_micros'])

                    filename = f"img_1_{ts}.jpg"
                    savePath = os.path.join(outputSubfolder, filename)

                    img = Image.open(io.BytesIO(imageBytes))
                    img.save(savePath, "JPEG")
            else:
                print(f"Warning: {filename} has no front-facing images.")

        except Exception as e:
            print(f"Error in {filename}: {e}")

    print(f"\n¡Done! All image sequences were saved to: {outputBaseDir}")
