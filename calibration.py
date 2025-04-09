

import cv2


import json

import numpy as np


def calibrate_piece_colors_with_browsing(video_cap, roi, fps, frame_count, start_frame):
    """Enhanced function to calibrate color ranges for each Tetris piece type with video browsing."""
    # Make sure we're at the right position in the video
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = video_cap.read()
    if not ret:
        print("Error: Could not read frame from video")
        return None

    # Extract the ROI
    x1, y1, x2, y2 = roi
    roi_img = frame[y1:y2, x1:x2]

    # Define the piece types and their expected colors
    piece_types = [
        ('I', 'Red'),
        ('T', 'Cyan'),
        ('L', 'Orange'),
        ('J', 'Blue'),
        ('S', 'Purple'),
        ('Z', 'Green'),
        ('O', 'Yellow'),
    ]

    piece_colors = {}

    # Create a window for color calibration with a fixed size
    window_name = 'Color Calibration with Video Browsing'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 700)  # Slightly larger to accommodate video controls

    # Define the text box height
    color_info_height = 150  # Increased to accommodate video browsing info

    # Variables for tracking position in video
    current_frame = start_frame
    current_second = current_frame / fps
    duration = frame_count / fps

    # Start calibrating each piece type
    for piece_idx, (piece_type, color_name) in enumerate(piece_types):
        # Display instructions
        print(f"\nCalibrating color for {piece_type} piece ({color_name}) - {piece_idx+1}/{len(piece_types)}:")
        print("1. Use WASD keys to browse the video and find a frame with the piece")
        print("2. Draw a rectangle on the piece to sample its color")
        print("3. Adjust trackbars if needed to refine the color range")
        print("4. Press 'c' to confirm and move to the next piece")
        print("5. Press 'r' to reset selection")
        print("6. Press 'esc' to cancel calibration")

        # Create trackbars for HSV range - grouped by H, S, V
        cv2.createTrackbar('H Lower', window_name, 0, 179, lambda x: None)
        cv2.createTrackbar('H Upper', window_name, 179, 179, lambda x: None)
        cv2.createTrackbar('S Lower', window_name, 0, 255, lambda x: None)
        cv2.createTrackbar('S Upper', window_name, 255, 255, lambda x: None)
        cv2.createTrackbar('V Lower', window_name, 0, 255, lambda x: None)
        cv2.createTrackbar('V Upper', window_name, 255, 255, lambda x: None)

        # Add a special checkbox for red hue wrap-around
        cv2.createTrackbar('Red wrap-around', window_name, 0, 1, lambda x: None)

        # Set initial values for common colors
        if color_name == 'Red':
            cv2.setTrackbarPos('H Lower', window_name, 160)
            cv2.setTrackbarPos('H Upper', window_name, 179)
            # Enable red wrap-around checkbox for red by default
            cv2.setTrackbarPos('Red wrap-around', window_name, 1)
        elif color_name == 'Cyan':
            cv2.setTrackbarPos('H Lower', window_name, 85)
            cv2.setTrackbarPos('H Upper', window_name, 95)
        elif color_name == 'Orange':
            cv2.setTrackbarPos('H Lower', window_name, 10)
            cv2.setTrackbarPos('H Upper', window_name, 25)
        elif color_name == 'Blue':
            cv2.setTrackbarPos('H Lower', window_name, 100)
            cv2.setTrackbarPos('H Upper', window_name, 140)
        elif color_name == 'Purple':
            cv2.setTrackbarPos('H Lower', window_name, 120)
            cv2.setTrackbarPos('H Upper', window_name, 150)
        elif color_name == 'Green':
            cv2.setTrackbarPos('H Lower', window_name, 45)
            cv2.setTrackbarPos('H Upper', window_name, 75)
        elif color_name == 'Yellow':
            cv2.setTrackbarPos('H Lower', window_name, 20)
            cv2.setTrackbarPos('H Upper', window_name, 40)

        # For all colors, start with moderate saturation and value
        cv2.setTrackbarPos('S Lower', window_name, 100)
        cv2.setTrackbarPos('S Upper', window_name, 255)
        cv2.setTrackbarPos('V Lower', window_name, 100)
        cv2.setTrackbarPos('V Upper', window_name, 255)

        # Variables for color sampling
        drawing = False
        sample_start = None
        sample_end = None

        # Create a copy of the ROI image for display
        display_roi = roi_img.copy()

        # Calculate scaling factor for display
        roi_height, roi_width = roi_img.shape[:2]
        max_width = 380
        scale_factor = min(max_width / roi_width, 300 / roi_height)

        new_roi_width = int(roi_width * scale_factor)
        new_roi_height = int(roi_height * scale_factor)

        def mouse_callback_color(event, x, y, flags, param):
            nonlocal drawing, sample_start, sample_end, display_roi, scale_factor

            # Correct for the color_info_height offset
            if y < color_info_height:
                return

            # Adjust y coordinate by subtracting the text box height
            y_adjusted = y - color_info_height

            # Convert display coordinates back to original image coordinates
            orig_x = int(x / scale_factor)
            orig_y = int(y_adjusted / scale_factor)

            # Make sure coordinates are within bounds of original image
            orig_x = max(0, min(orig_x, roi_width - 1))
            orig_y = max(0, min(orig_y, roi_height - 1))

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                sample_start = (orig_x, orig_y)
                display_roi = roi_img.copy()

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    temp_img = roi_img.copy()
                    # Draw rectangle using original coordinates
                    cv2.rectangle(temp_img, sample_start, (orig_x, orig_y), (0, 255, 0), 2)
                    display_roi = temp_img

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                sample_end = (orig_x, orig_y)

                # Draw the rectangle on the display image
                cv2.rectangle(display_roi, sample_start, sample_end, (0, 255, 0), 2)

                # Extract the sample region
                x1, y1 = min(sample_start[0], sample_end[0]), min(sample_start[1], sample_end[1])
                x2, y2 = max(sample_start[0], sample_end[0]), max(sample_start[1], sample_end[1])

                # Make sure coordinates are within bounds
                x1 = max(0, min(x1, roi_img.shape[1]-1))
                y1 = max(0, min(y1, roi_img.shape[0]-1))
                x2 = max(0, min(x2, roi_img.shape[1]-1))
                y2 = max(0, min(y2, roi_img.shape[0]-1))

                # Sample colors from the region
                if x1 < x2 and y1 < y2:
                    sample_region = roi_img[y1:y2, x1:x2]
                    if sample_region.size > 0:
                        # Convert to HSV
                        hsv_sample = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)

                        # Calculate min and max HSV values
                        h_min = np.percentile(hsv_sample[:,:,0], 5)
                        h_max = np.percentile(hsv_sample[:,:,0], 95)
                        s_min = np.percentile(hsv_sample[:,:,1], 5)
                        s_max = np.percentile(hsv_sample[:,:,1], 95)
                        v_min = np.percentile(hsv_sample[:,:,2], 5)
                        v_max = np.percentile(hsv_sample[:,:,2], 95)

                        # For red detection, check if the hue is in the red range
                        is_red = (h_min < 10 or h_max > 170)
                        if is_red:
                            # Set the red wrap-around checkbox
                            cv2.setTrackbarPos('Red wrap-around', window_name, 1)

                        # Add some margin
                        h_min = max(0, h_min - 10)
                        h_max = min(179, h_max + 10)
                        s_min = max(0, s_min - 40)
                        s_max = min(255, s_max + 40)
                        v_min = max(0, v_min - 40)
                        v_max = min(255, v_max + 40)

                        # Update trackbars
                        cv2.setTrackbarPos('H Lower', window_name, int(h_min))
                        cv2.setTrackbarPos('H Upper', window_name, int(h_max))
                        cv2.setTrackbarPos('S Lower', window_name, int(s_min))
                        cv2.setTrackbarPos('S Upper', window_name, int(s_max))
                        cv2.setTrackbarPos('V Lower', window_name, int(v_min))
                        cv2.setTrackbarPos('V Upper', window_name, int(v_max))

        # Set mouse callback for color sampling
        cv2.setMouseCallback(window_name, mouse_callback_color)

        # Main loop for this piece type calibration
        while True:
            # Get current positions of trackbars
            h_lower = cv2.getTrackbarPos('H Lower', window_name)
            h_upper = cv2.getTrackbarPos('H Upper', window_name)
            s_lower = cv2.getTrackbarPos('S Lower', window_name)
            s_upper = cv2.getTrackbarPos('S Upper', window_name)
            v_lower = cv2.getTrackbarPos('V Lower', window_name)
            v_upper = cv2.getTrackbarPos('V Upper', window_name)
            is_red_wrap = cv2.getTrackbarPos('Red wrap-around', window_name) == 1

            # Convert ROI to HSV
            hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

            # Create a mask using the selected HSV range
            if is_red_wrap:
                # For red hue that wraps around: create two separate masks and combine them
                lower_bound1 = np.array([0, s_lower, v_lower])
                upper_bound1 = np.array([10, s_upper, v_upper])
                mask1 = cv2.inRange(hsv_roi, lower_bound1, upper_bound1)

                lower_bound2 = np.array([160, s_lower, v_lower])
                upper_bound2 = np.array([179, s_upper, v_upper])
                mask2 = cv2.inRange(hsv_roi, lower_bound2, upper_bound2)

                mask = cv2.bitwise_or(mask1, mask2)
            else:
                # Normal case: single hue range
                lower_bound = np.array([h_lower, s_lower, v_lower])
                upper_bound = np.array([h_upper, s_upper, v_upper])
                mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)

            # Apply the mask to the ROI
            result = cv2.bitwise_and(roi_img, roi_img, mask=mask)

            # Resize images to maintain aspect ratio
            display_roi_scaled = cv2.resize(display_roi, (new_roi_width, new_roi_height))
            mask_scaled = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (new_roi_width, new_roi_height))
            result_scaled = cv2.resize(result, (new_roi_width, new_roi_height))

            # Calculate total width needed
            total_width = new_roi_width * 3 + 20  # 3 images with 10px spacing between them

            # Create a dark background
            background = np.zeros((max(new_roi_height, 100) + 50, total_width, 3), dtype=np.uint8)

            # Place images on the background
            background[0:new_roi_height, 0:new_roi_width] = display_roi_scaled
            background[0:new_roi_height, new_roi_width+10:new_roi_width*2+10] = mask_scaled
            background[0:new_roi_height, new_roi_width*2+20:new_roi_width*3+20] = result_scaled

            # Add labels
            cv2.putText(background, "Original ROI", (10, new_roi_height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(background, "Mask", (new_roi_width+20, new_roi_height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(background, "Result", (new_roi_width*2+30, new_roi_height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Create a color info banner with the SAME WIDTH as the background
            color_info = np.zeros((color_info_height, total_width, 3), dtype=np.uint8)

            # Add title and current piece being calibrated
            cv2.putText(color_info, f"Calibrating {piece_type} piece ({color_name}) - {piece_idx+1}/{len(piece_types)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show HSV values and red wrap-around status
            if is_red_wrap:
                hsv_text = f"H: 0-10 & 160-179  S: {s_lower}-{s_upper}  V: {v_lower}-{v_upper} (Red wrap-around: ON)"
            else:
                hsv_text = f"H: {h_lower}-{h_upper}  S: {s_lower}-{s_upper}  V: {v_lower}-{v_upper}"

            cv2.putText(color_info, hsv_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add video position info
            video_text = f"Time: {current_second:.2f}s / {duration:.2f}s | Frame: {current_frame} / {frame_count}"
            cv2.putText(color_info, video_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add controls reminder
            controls_text = "W/S: ±10s | A/D: ±1s | Q/E: ±1 frame | R: Reset | C: Confirm"
            cv2.putText(color_info, controls_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Create the final display image by stacking the color info and background
            final_display = np.vstack((color_info, background))

            # Show the final display
            cv2.imshow(window_name, final_display)

            key = cv2.waitKey(30) & 0xFF

            # Handle keyboard inputs for video browsing and color confirmation
            if key == 27:  # ESC key
                print("Calibration cancelled.")
                return None
            elif key == ord('c'):
                # Confirm and save the color range
                color_data = {
                    'h_lower': h_lower,
                    'h_upper': h_upper,
                    's_lower': s_lower,
                    's_upper': s_upper,
                    'v_lower': v_lower,
                    'v_upper': v_upper,
                    'is_red': is_red_wrap  # Store the red wrap-around flag
                }
                piece_colors[piece_type] = color_data
                print(f"Color range for {piece_type} piece saved.")
                break
            elif key == ord('r'):
                # Reset trackbars and display
                display_roi = roi_img.copy()
                print("Selection and trackbars reset.")
            elif key == ord('w'):  # Jump forward 10 seconds
                current_second = min(duration, current_second + 10)
                current_frame = int(current_second * fps)
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = video_cap.read()
                if ret:
                    roi_img = frame[y1:y2, x1:x2]
                    display_roi = roi_img.copy()
                else:
                    print("End of video reached.")
                    current_frame = min(current_frame, frame_count - 1)
                    current_second = current_frame / fps
            elif key == ord('s'):  # Jump backward 10 seconds
                current_second = max(0, current_second - 10)
                current_frame = int(current_second * fps)
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = video_cap.read()
                if ret:
                    roi_img = frame[y1:y2, x1:x2]
                    display_roi = roi_img.copy()
            elif key == ord('d'):  # Step forward 1 second
                current_second = min(duration, current_second + 1)
                current_frame = int(current_second * fps)
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = video_cap.read()
                if ret:
                    roi_img = frame[y1:y2, x1:x2]
                    display_roi = roi_img.copy()
                else:
                    print("End of video reached.")
                    current_frame = min(current_frame, frame_count - 1)
                    current_second = current_frame / fps
            elif key == ord('a'):  # Step backward 1 second
                current_second = max(0, current_second - 1)
                current_frame = int(current_second * fps)
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = video_cap.read()
                if ret:
                    roi_img = frame[y1:y2, x1:x2]
                    display_roi = roi_img.copy()
            elif key == ord('e'):  # Step forward 1 frame
                current_frame = min(frame_count - 1, current_frame + 1)
                current_second = current_frame / fps
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = video_cap.read()
                if ret:
                    roi_img = frame[y1:y2, x1:x2]
                    display_roi = roi_img.copy()
                else:
                    print("End of video reached.")
                    current_frame = min(current_frame, frame_count - 1)
                    current_second = current_frame / fps
            elif key == ord('q'):  # Step backward 1 frame
                current_frame = max(0, current_frame - 1)
                current_second = current_frame / fps
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = video_cap.read()
                if ret:
                    roi_img = frame
                    roi_img = frame[y1:y2, x1:x2]
                    display_roi = roi_img.copy()

     # Return the collected piece_colors data
    return piece_colors
    print("end of calibration")


def run_calibration(video_path):
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
    print(f"FPS: {fps:.2f}")

    # Initialize current position to 5 seconds into the video (to avoid intros)
    current_second = 5
    current_frame = int(current_second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    # Create a window
    window_name = 'Tetris Calibration - Video Browser'
    cv2.namedWindow(window_name)

    # Variables for ROI selection
    roi = None
    drawing = False
    start_point = None
    end_point = None

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video")
        cap.release()
        return

    # Create a copy of the frame to draw on
    display_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, display_frame, frame, roi

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            display_frame = frame.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                display_frame = frame.copy()
                cv2.rectangle(display_frame, start_point, (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            cv2.rectangle(display_frame, start_point, end_point, (0, 255, 0), 2)

            # Extract ROI coordinates
            x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
            x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
            roi = (x1, y1, x2, y2)

            print(f"Selected ROI: {roi}")

    # Set mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\nVideo Browsing Controls:")
    print("W - Jump forward 10 seconds")
    print("S - Jump backward 10 seconds")
    print("D - Step forward 1 second")
    print("A - Step backward 1 second")
    print("E - Step forward 1 frame")
    print("Q - Step backward 1 frame")
    print("R - Reset ROI selection")
    print("C - Confirm ROI selection and proceed to color calibration")
    print("ESC - Quit")

    # Main browsing and selection loop
    while True:
        # Create info text to display
        info_text = f"Time: {current_second:.2f}s / {duration:.2f}s | Frame: {current_frame} / {frame_count}"
        controls_text = "W/S: ±10s | A/D: ±1s | Q/E: ±1 frame | R: Reset | C: Confirm"

        # Create a copy for display
        info_frame = display_frame.copy()

        # Add text overlay
        cv2.putText(info_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_frame, controls_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # If ROI is selected, draw it
        if roi:
            cv2.rectangle(info_frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
            cv2.putText(info_frame, "ROI Selected", (roi[0], roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow(window_name, info_frame)

        # Wait for keyboard input
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC key
            break
        elif key == ord('w'):  # Jump forward 10 seconds
            current_second = min(duration, current_second + 10)
            current_frame = int(current_second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if ret:
                display_frame = frame.copy()
            else:
                print("End of video reached.")
                current_frame = min(current_frame, frame_count - 1)
                current_second = current_frame / fps
        elif key == ord('s'):  # Jump backward 10 seconds
            current_second = max(0, current_second - 10)
            current_frame = int(current_second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if ret:
                display_frame = frame.copy()
        elif key == ord('d'):  # Step forward 1 second
            current_second = min(duration, current_second + 1)
            current_frame = int(current_second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if ret:
                display_frame = frame.copy()
            else:
                print("End of video reached.")
                current_frame = min(current_frame, frame_count - 1)
                current_second = current_frame / fps
        elif key == ord('a'):  # Step backward 1 second
            current_second = max(0, current_second - 1)
            current_frame = int(current_second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if ret:
                display_frame = frame.copy()
        elif key == ord('e'):  # Step forward 1 frame
            current_frame = min(frame_count - 1, current_frame + 1)
            current_second = current_frame / fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if ret:
                display_frame = frame.copy()
            else:
                print("End of video reached.")
                current_frame = min(current_frame, frame_count - 1)
                current_second = current_frame / fps
        elif key == ord('q'):  # Step backward 1 frame
            current_frame = max(0, current_frame - 1)
            current_second = current_frame / fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if ret:
                display_frame = frame.copy()
        elif key == ord('r'):  # Reset ROI
            display_frame = frame.copy()
            roi = None
            print("ROI selection reset.")
        elif key == ord('c') and roi:  # Confirm selection
            print("ROI selection confirmed!")
            break

    # If we have a valid ROI, proceed with color calibration
    calibration_data = {}
    if roi:
        calibration_data['roi'] = roi

        # Now we continue to the color calibration phase
        # but with video browsing capabilities too
        color_calibration_result = calibrate_piece_colors_with_browsing(cap, roi, fps, frame_count, current_frame)


        if color_calibration_result:
            calibration_data['piece_colors'] = color_calibration_result

            # Save calibration data to file
            with open('calibration.json', 'w') as f:
                json.dump(calibration_data, f)
            print(f"Calibration data saved to calibration.json")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()