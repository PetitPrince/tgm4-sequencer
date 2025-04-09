# TGM4 sequencer

Get pieces from footage of TGM4 and output it in a text file. Project was mostly vibed with Claude 3.7.

# Usage
I'm using UV, so running it with it should automagically install the dependencies (OpenCV, NumPy)
```sh
uv run main.py
```

## Calibration Mode
First, calibrate the color detection for your video:

During calibration:

1. Use WASD keys to navigate the video and find a good frame
2. Draw a rectangle around the piece preview area
3. Press 'C' to confirm ROI selection
4. For each piece type:
  * Navigate to a frame showing the piece
  * Draw a rectangle around the piece to sample its color
  * Adjust HSV sliders if needed
  * Press 'C' to confirm and move to next piece
  * Press 'R' to reset selection
  * Press 'ESC' to cancel

Default piece color are set to TGM, but you can calibrate for standard and it should work as well.

## Detection Mode
After calibration, run the piece detection:

 ### Optional flags:

* --debug: Save debug frames showing piece detection
--no-display: Run without visualization window (faster)

### Files generated
* calibration.json: Stores ROI coordinates and HSV color ranges
* *_pieces.txt: Output file containing detected piece sequence