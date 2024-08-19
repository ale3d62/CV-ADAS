import sys

def printProgress(iImg, nImg):
    bar_length = 40 
    progress = (iImg + 1) / nImg 
    block = int(round(bar_length * progress))
    
    progress_text = f"\rRunning Benchmark {iImg + 1}/{nImg} [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.1f}%"
    sys.stdout.write(progress_text)
    sys.stdout.flush()