import os
import subprocess
import glob


def create_videos_from_images(inputDir, outputDir, fps=10):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    sequenceFolders = [d for d in glob.glob(os.path.join(inputDir, "*")) if os.path.isdir(d)]

    for folder in sequenceFolders:
        sequenceName = os.path.basename(folder)
        videoOutput = os.path.join(outputDir, f"{sequenceName}.mp4")
        list_txt = os.path.join(folder, "input_list.txt")

        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))

        if not images:
            continue

        with open(list_txt, 'w') as f:
            for img_path in images:
                f.write(f"file '{os.path.abspath(img_path)}'\n")

        comando = [
            'ffmpeg', '-y',
            '-r', str(fps),
            '-f', 'concat',
            '-safe', '0',
            '-i', list_txt,
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '17',
            '-pix_fmt', 'yuv420p',
            videoOutput
        ]

        try:
            print(f"Generating video for sequence: {sequenceName}...")
            resultado = subprocess.run(comando, capture_output=True, text=True)

            #Delete temp list txt file
            if os.path.exists(list_txt):
                os.remove(list_txt)

            if resultado.returncode != 0:
                print(f"Error in sequence {sequenceName}: {resultado.stderr}")

        except Exception as e:
            print(f"Error: {e}")
