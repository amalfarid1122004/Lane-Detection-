import cv2
import numpy as np
import os
import shutil

def process_video(input_path):
    # Check if the input file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    # Create output directories
    base_dir = os.path.dirname(input_path)
    output_dir = os.path.join(base_dir, "E:/algoAsignments/[TEMPLATE]/ouput_frames_lane_detection")
    
    # Create folders for each processing step
    steps_folders = [
        "1_original_frames", 
        "2_hsv_conversion",
        "3_grayscale", 
        "4_yellow_mask", 
        "5_white_mask", 
        "6_combined_mask", 
        "7_denoised_mask", 
        "8_edges", 
        "9_roi_mask", 
        "10_masked_edges", 
        "11_final_result"
    ]
    
    # Remove existing output folder if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create new output folder and subfolders
    os.makedirs(output_dir)
    for folder in steps_folders:
        os.makedirs(os.path.join(output_dir, folder))

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output video writer
    output_video_path = os.path.join(output_dir, 'output_lane_detection.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Previous lane lines for smoothing
    left_line_history = []
    right_line_history = []
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to detect lanes and get intermediate results
        processed_frame, intermediate_results = detect_lane_lines_with_steps(frame, left_line_history, right_line_history)
        
        # Save each processing step for this frame
        if frame_count % 10 == 0:  # Save every 10th frame to avoid too many images
            # Save original frame
            cv2.imwrite(os.path.join(output_dir, "1_original_frames", f"frame_{frame_count:04d}.jpg"), frame)
            
            # Save intermediate results
            for step_name, step_result in intermediate_results.items():
                # Convert grayscale images to BGR for consistent saving
                if len(step_result.shape) == 2:
                    step_result_colored = cv2.cvtColor(step_result, cv2.COLOR_GRAY2BGR)
                else:
                    step_result_colored = step_result
                    
                cv2.imwrite(os.path.join(output_dir, step_name, f"frame_{frame_count:04d}.jpg"), step_result_colored)
            
            # Save final result
            cv2.imwrite(os.path.join(output_dir, "11_final_result", f"frame_{frame_count:04d}.jpg"), processed_frame)
        
        # Display the result
        cv2.imshow('Lane Detection', processed_frame)
        
        # Write to output video
        out.write(processed_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to {output_dir}")
    print(f"Output video saved to {output_video_path}")

def detect_lane_lines_with_steps(frame, left_line_history, right_line_history):
    intermediate_results = {}
    
    # Create a copy of the frame for drawing
    result_frame = frame.copy()
    
    # Step 1: RGB to HSV conversion
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    intermediate_results["2_hsv_conversion"] = hsv
    
    # Step 2: RGB to Grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    intermediate_results["3_grayscale"] = gray
    
    # Step 3: Create masks for yellow and white lanes
    # Mask for yellow lane lines
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    intermediate_results["4_yellow_mask"] = yellow_mask
    
    # Mask for white lane lines
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    intermediate_results["5_white_mask"] = white_mask
    
    # Combine masks
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    intermediate_results["6_combined_mask"] = combined_mask
    
    # Step 4: Remove noise
    kernel = np.ones((5, 5), np.uint8)
    denoised_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    denoised_mask = cv2.morphologyEx(denoised_mask, cv2.MORPH_OPEN, kernel)
    intermediate_results["7_denoised_mask"] = denoised_mask
    
    # Step 5: Edge detection using Canny
    edges = cv2.Canny(denoised_mask, 50, 150)
    intermediate_results["8_edges"] = edges
    
    # Step 6: Define region of interest (ROI)
    height, width = edges.shape
    roi_vertices = np.array([
        [(0, height), (width // 2 - 50, height // 2 + 50), 
         (width // 2 + 50, height // 2 + 50), (width, height)]
    ], dtype=np.int32)
    
    # Create a mask for ROI
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    intermediate_results["9_roi_mask"] = roi_mask
    
    # Apply ROI mask
    masked_edges = cv2.bitwise_and(edges, roi_mask)
    intermediate_results["10_masked_edges"] = masked_edges
    
    # Step 7: Hough Transform to detect lines
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=20,
        maxLineGap=300
    )
    
    # Step 8: Process the detected lines to identify left and right lanes
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope
            if x2 - x1 == 0:  # Avoid division by zero
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter lines based on slope
            if abs(slope) < 0.3:  # Ignore horizontal lines
                continue
                
            # Separate left and right lane lines based on slope
            if slope < 0:  # Left lane has negative slope
                left_lines.append(line[0])
            else:  # Right lane has positive slope
                right_lines.append(line[0])
    
    # Step 9: Average and extrapolate the lane lines
    left_line = average_lane_line(left_lines, frame.shape[0], left_line_history)
    right_line = average_lane_line(right_lines, frame.shape[0], right_line_history)
    
    # Step 10: Draw lanes on the original frame
    if left_line is not None:
        cv2.line(result_frame, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 5)
        left_line_history.append(left_line)
        if len(left_line_history) > 10:
            left_line_history.pop(0)
    
    if right_line is not None:
        cv2.line(result_frame, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 5)
        right_line_history.append(right_line)
        if len(right_line_history) > 10:
            right_line_history.pop(0)
    
    return result_frame, intermediate_results

def average_lane_line(lines, y_max, line_history):
    if not lines and not line_history:
        return None
    
    # If no lines detected in this frame but we have history
    if not lines and line_history:
        # Return the last known line
        return line_history[-1]
    
    # Calculate average slope and intercept
    slopes = []
    intercepts = []
    
    for line in lines:
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:  # Avoid division by zero
            continue
            
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        slopes.append(slope)
        intercepts.append(intercept)
    
    if not slopes or not intercepts:
        if line_history:
            return line_history[-1]
        return None
    
    # Average values
    avg_slope = sum(slopes) / len(slopes)
    avg_intercept = sum(intercepts) / len(intercepts)
    
    # Use smoothing if we have history
    if line_history:
        prev_line = line_history[-1]
        prev_slope = (prev_line[3] - prev_line[1]) / (prev_line[2] - prev_line[0]) if prev_line[2] != prev_line[0] else 0
        prev_intercept = prev_line[1] - prev_slope * prev_line[0]
        
        # Smoothing factor (0.8 means 80% of the new value, 20% of the old value)
        smoothing_factor = 0.8
        avg_slope = smoothing_factor * avg_slope + (1 - smoothing_factor) * prev_slope
        avg_intercept = smoothing_factor * avg_intercept + (1 - smoothing_factor) * prev_intercept
    
    # Calculate line endpoints
    y1 = y_max  # Bottom of the image
    y2 = int(y_max * 0.6)  # A bit above the middle
    
    x1 = int((y1 - avg_intercept) / avg_slope) if avg_slope != 0 else 0
    x2 = int((y2 - avg_intercept) / avg_slope) if avg_slope != 0 else 0
    
    return [x1, y1, x2, y2]

if __name__ == "__main__":
    input_path = "C:/Users/fared/Downloads/Telegram Desktop/video_2025-05-12_21-41-19.mp4"
    process_video(input_path)