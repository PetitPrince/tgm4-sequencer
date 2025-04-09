import cv2
import numpy as np
import argparse
import json
import os
import time
from datetime import timedelta

from calibration import run_calibration


def save_debug_frame(frame, frame_idx, timestamp, piece_type, output_dir):
    """Save a debug frame with detection information."""
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    global debug_frame_index
    padding = 3 
    # Format timestamp as MM_SS
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    
    # Create filename with frame number and timestamp
    filename = f"{debug_frame_index}_frame_{frame_idx:06d}_time_{minutes:02d}_{seconds:02d}_piece_{piece_type}_{debug_frame_index:0{padding}d}.jpg"
    filepath = os.path.join(output_dir, filename)
    debug_frame_index += 1
    # Save the frame
    cv2.imwrite(filepath, frame)
    print(f"Debug frame saved: {filepath}")


def run_detection(video_path, debug=False, display=True):
    """Function to detect Tetris pieces from a video using calibration data and grid approach."""
    global debug_frame_index
    debug_frame_index = 1
    # Load calibration data
    try:
        with open('calibration.json', 'r') as f:
            calibration_data = json.load(f)
    except FileNotFoundError:
        print("Error: Calibration data not found. Please run calibration first.")
        return
    
    # Extract calibration information
    roi = tuple(calibration_data['roi'])
    piece_colors = calibration_data['piece_colors']
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Total frames: {frame_count}")
    
    # Create debug directory if debug mode is enabled
    debug_dir = None
    if debug:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        debug_dir = f"{video_name}_debug_frames"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        print(f"Debug mode enabled. Frames will be saved to: {debug_dir}")
    
    # Create a window for visualization only if display is enabled
    window_name = 'Tetris Piece Detection'
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        print("Display mode enabled. Showing detection window.")
    else:
        print("Display mode disabled. Running in headless mode for faster processing.")
    
    # Variables for piece detection
    piece_sequence = []
    previous_grid_pieces = []  # To store the previous grid state
    frame_skip = max(1, int(fps / 30))  # Process 10 frames per second
    
    # Number of preview cells to look for
    num_preview_cells = 5  # From your description - 5 preview pieces in vertical box
    
    # Create output file
    output_file = os.path.splitext(video_path)[0] + "_pieces.txt"
    # Clear the output file if it exists
    with open(output_file, 'w') as f:
        pass
    
    # Initialize progress tracking
    start_time = time.time()
    progress_interval = max(1, int(frame_count / 100))  # Update progress every 1%
    
    print("Starting piece detection...")
    
    # Process video frame by frame
    frame_idx = 0
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate current timestamp in seconds
        current_timestamp = frame_idx / fps
        
        # Process only every Nth frame to speed up detection
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        # Extract ROI
        x1, y1, x2, y2 = roi
        roi_img = frame[y1:y2, x1:x2]
        
        # Calculate grid cell dimensions
        roi_height, roi_width = roi_img.shape[:2]
        cell_height = roi_height / num_preview_cells
        
        # Create a display copy of the ROI (only if needed for display or debug)
        if display or debug:
            display_roi = roi_img.copy()
            diff_display = np.zeros_like(display_roi)
        
        # List to store the current grid pieces
        current_grid_pieces = []
        
        # Process each grid cell
        for i in range(num_preview_cells):
            # Calculate cell boundaries
            cell_y1 = int(i * cell_height)
            cell_y2 = int((i + 1) * cell_height)
            
            # Extract the cell
            cell_img = roi_img[cell_y1:cell_y2, :]
            
            # Convert cell to HSV
            hsv_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
            
            # Dictionary to store the color match scores for each piece type
            piece_scores = {}
            piece_masks = {}
            
            # Check each piece type's color match
            for piece_type, color_info in piece_colors.items():
                # Create HSV range
                lower_bound = np.array([color_info['h_lower'], color_info['s_lower'], color_info['v_lower']])
                upper_bound = np.array([color_info['h_upper'], color_info['s_upper'], color_info['v_upper']])
                
                # Special case for red (spanning both ends of the hue spectrum)
                if color_info.get('is_red', False):
                    # Create two masks: one for low red (0-10) and one for high red (160-179)
                    mask1 = cv2.inRange(hsv_cell, 
                                    np.array([0, color_info['s_lower'], color_info['v_lower']]), 
                                    np.array([10, color_info['s_upper'], color_info['v_upper']]))
                    mask2 = cv2.inRange(hsv_cell, 
                                    np.array([160, color_info['s_lower'], color_info['v_lower']]), 
                                    np.array([179, color_info['s_upper'], color_info['v_upper']]))
                    # Combine the masks
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    # Create normal mask for other pieces
                    mask = cv2.inRange(hsv_cell, lower_bound, upper_bound)
                
                # Apply morphological operations to clean up the mask
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Calculate match score (percentage of pixels that match the color)
                match_score = np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255)
                piece_scores[piece_type] = match_score
                piece_masks[piece_type] = mask
            
            # Find the piece with the highest match score
            best_piece = max(piece_scores.items(), key=lambda x: x[1]) if piece_scores else (None, 0)
            piece_type, score = best_piece
            
            # Define a threshold for accepting a match
            match_threshold = 0.05  # Adjust this threshold as needed
            
            if score > match_threshold:
                # Add the detected piece to the current grid
                current_grid_pieces.append(piece_type)
                
                # Draw cell boundaries and piece type (only if needed for display or debug)
                if display or debug:
                    cv2.rectangle(display_roi, (0, cell_y1), (roi_width, cell_y2), (0, 255, 0), 2)
                    cv2.putText(display_roi, piece_type, (10, cell_y1 + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the best mask on the diff display if this piece is new or moved
                if (display or debug) and (i >= len(previous_grid_pieces) or piece_type != previous_grid_pieces[i]):
                    # This piece is new or different from previous frame
                    # Create a colored version of the mask for visualization
                    if piece_type:
                        best_mask = piece_masks[piece_type]
                        diff_cell = cv2.bitwise_and(cell_img, cell_img, mask=best_mask)
                        diff_display[cell_y1:cell_y2, :] = diff_cell
                        
                        # Add label to diff display
                        cv2.putText(diff_display, f"NEW {piece_type}", (10, cell_y1 + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # No piece detected in this cell
                current_grid_pieces.append(None)
                
                # Draw cell boundaries with red to indicate no piece detected (only if needed for display or debug)
                if display or debug:
                    cv2.rectangle(display_roi, (0, cell_y1), (roi_width, cell_y2), (0, 0, 255), 2)
        
        # Check if the grid has changed
        grid_changed = False
        new_pieces = []
        
        # Check if there are enough valid pieces detected in the current frame
        valid_pieces = [p for p in current_grid_pieces if p is not None]
        
        if not previous_grid_pieces and valid_pieces:
            # First frame with detected pieces - take all valid pieces as new
            grid_changed = True
            new_pieces = valid_pieces
            print(f"Frame {frame_idx}: First detection - found {len(new_pieces)} pieces")
        elif previous_grid_pieces and valid_pieces:
            # Compare entire grid with previous state
            # If the pieces have shifted up (indicating a new piece was added)
            
            # Get valid pieces from previous frame
            prev_valid_pieces = [p for p in previous_grid_pieces if p is not None]
            
            # Check if the preview stack has shifted (pieces moved up)
            if len(valid_pieces) >= len(prev_valid_pieces):
                # Determine if the entire preview has shifted
                shifted = False
                
                # Check if the pieces have shifted up one position
                if len(prev_valid_pieces) > 0 and valid_pieces[:-1] == prev_valid_pieces[1:]:
                    shifted = True
                    new_piece = valid_pieces[-1]  # The newest piece (at the bottom)
                    new_pieces.append(new_piece)
                    grid_changed = True
        
        # If grid changed, add new pieces to sequence and save debug frame if enabled
        if grid_changed:
            # Prepare visualization (only if needed for display or debug)
            if display or debug:
                # Combine display and diff into one window for visualization
                h_stack = np.hstack((display_roi, diff_display)) if display_roi.shape[1] == diff_display.shape[1] else display_roi
            
            # Add new pieces to the piece sequence
            for piece in new_pieces:
                piece_sequence.append(piece)
                print(f"Frame {frame_idx}: Detected new piece: {piece}")
                
                # Save debug frame if debug mode is enabled
                if debug and debug_dir:
                    # Create full frame with detection information
                    full_debug_frame = frame.copy()
                    cv2.rectangle(full_debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Place the h_stack visualization on the frame
                    if y1 + h_stack.shape[0] <= full_debug_frame.shape[0] and x1 + h_stack.shape[1] <= full_debug_frame.shape[1]:
                        full_debug_frame[y1:y1+h_stack.shape[0], x1:x1+h_stack.shape[1]] = h_stack
                    
                    # Add frame info and piece info as text
                    cv2.putText(full_debug_frame, f"Frame: {frame_idx} | Time: {current_timestamp:.2f}s | Piece: {piece}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save the debug frame
                    save_debug_frame(full_debug_frame, frame_idx, current_timestamp, piece, debug_dir)
                
                # Save piece to output file
                with open(output_file, 'a') as f:
                    f.write(f"{piece}\n")
                # time.sleep(2)
                
        # Update previous grid pieces
        previous_grid_pieces = current_grid_pieces.copy()
        
        # Show detection results if display is enabled
        if display:
            # Combine display and diff into one window
            h_stack = np.hstack((display_roi, diff_display)) if display_roi.shape[1] == diff_display.shape[1] else display_roi
            cv2.imshow(window_name, h_stack)
            
            # Check for keypress to exit early
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Update progress bar
        if frame_idx % progress_interval == 0:
            progress = frame_idx / frame_count * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / (progress / 100) if progress > 0 else 0
            remaining_time = estimated_total - elapsed_time
            
            print(f"Progress: {progress:.1f}% | Elapsed: {elapsed_time:.1f}s | Remaining: {remaining_time:.1f}s")
        
        frame_idx += 1
    
    # Clean up
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    # Print summary
    print(f"\nDetection complete!")
    print(f"Total pieces detected: {len(piece_sequence)}")
    print(f"Results saved to: {output_file}")
    
    # Print piece frequency analysis
    piece_counts = {}
    for piece in piece_sequence:
        if piece not in piece_counts:
            piece_counts[piece] = 0
        piece_counts[piece] += 1
    
    print("\nPiece frequency analysis:")
    total_pieces = len(piece_sequence)
    for piece, count in piece_counts.items():
        percentage = (count / total_pieces) * 100 if total_pieces > 0 else 0
        print(f"{piece}: {count} ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Tetris Piece Detector')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration mode')
    parser.add_argument('--detect', action='store_true', help='Run detection mode')
    parser.add_argument('--debug', action='store_true', help='Save debug frames with piece detections')
    parser.add_argument('--no-display', action='store_true', help='Run without displaying the detection window (faster)')
    args = parser.parse_args()

    if args.calibrate:
        run_calibration(args.video_path)
    elif args.detect:
        run_detection(args.video_path, debug=args.debug, display=not args.no_display)
    else:
        print("Please specify either --calibrate or --detect mode.")

if __name__ == "__main__":
    main()