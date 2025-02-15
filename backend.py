import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import pandas as pd
import re
import time
import zipfile

from datetime import datetime
from scipy.signal import savgol_filter
from simplification.cutil import simplify_coords
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

# preferences
SHOW_VISUALIZATION = False
SCALE_YAW = 1
THRESHOLD_YAW = 0.5
DISTANCE_TO_WALL = 0.5

BASE_DIR = "/home/superlinkfour/crackfinder_webapp"

# global variables
x_flight_path_origin = 0
y_flight_path_origin = 0
    
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)
        end_time = time.time()  # End the timer
        execution_time = end_time - start_time
        # ANSI escape codes for green and bold text
        print(f"\n\033[1;32mFunction '{func.__name__}' executed in {execution_time:.4f} seconds\033[0m.\n")
        return result
    return wrapper

@measure_time
def DataPreparation():

    # load data from uploads
    mycsv = ''
    for filename in os.listdir('uploads'):
        if filename.lower().endswith('.csv'):
            mycsv = filename
    mycsv = os.path.join(BASE_DIR,'uploads', mycsv)
    df_flowdeckdata = pd.read_csv(mycsv)
    x_coords = df_flowdeckdata['Y'].tolist()
    y_coords = df_flowdeckdata['X'].tolist()
    z_coords = df_flowdeckdata['Z'].tolist()
    yaws = df_flowdeckdata['Yaw'].tolist()
    timestamps = df_flowdeckdata['Timestamp'].tolist()

    # preprocessing of data
    x_coords = [-x for x in x_coords]
    yaws = [SCALE_YAW*yaw for yaw in yaws]
    minz = min(z_coords)
    z_coords = [z-minz for z in z_coords]

    # add dead time to the beginning
    x_coords = [0] * 200 + x_coords
    y_coords = [0] * 200 + y_coords
    z_coords = [0] * 200 + z_coords
    yaws = [0] * 200 + yaws
    timestamps = ['25/09/2024  12:17:12 pm'] * 200 + timestamps

    # update flowdeck data
    df_flowdeckdata_updated = pd.DataFrame()
    df_flowdeckdata_updated['X'] = x_coords
    df_flowdeckdata_updated['Y'] = y_coords
    df_flowdeckdata_updated['Z'] = z_coords
    df_flowdeckdata_updated['Yaw'] = yaws
    df_flowdeckdata_updated['Timestamp'] = timestamps
    
    # determine filename of zip file
    myzip = ''
    for filename in os.listdir('uploads'):
        if filename.lower().endswith('.zip'):
            myzip = filename
    myzip = os.path.join(BASE_DIR,'uploads', myzip)

    # preallocate for image data dataframe
    df_imagedata = pd.DataFrame()

    # extract images and times of capture
    with zipfile.ZipFile(myzip, 'r') as file:
        # pre allocate for filenames and datetime
        filenames = [''] * len(file.infolist())
        filetimes = [''] * len(file.infolist())
        for i, file_info in enumerate(file.infolist()):
            # check if file is an image file
            image_extensions = {'.dng', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
            isimage = False
            for extension in image_extensions:
                if file_info.filename.lower().endswith(extension):
                    isimage = True
                    break
            if isimage:
                # get filename
                filenames[i] = file_info.filename
                # get datetime
                filetimes[i] = datetime.fromtimestamp(time.mktime(file_info.date_time + (0, 0, 0))).strftime("%Y-%m-%d %H:%M:%S").replace(" 0", " ")

        # store image filename and datetime in dataframe
        df_imagedata = pd.DataFrame({'ImagePath': filenames, 'Timestamp': filetimes})
        df_imagedata.replace('', np.nan, inplace=True)
        df_imagedata = df_imagedata.dropna().reset_index(drop=True)
    
    # get the date of the survey
    dates = df_imagedata['Timestamp'].tolist()
    dt = datetime.strptime(dates[0], "%Y-%m-%d %H:%M:%S")
    year = str(dt.year)
    month = str(dt.month)
    day = str(dt.day)
    if len(month) == 1: month = '0' + month
    if len(day) == 1: day = '0' + day
    index = str(sum(1 for file in os.listdir('static') if file.startswith(f"rawloc_{year}_{month}_{day}")))

    # generate session id
    id = f"{year}_{month}_{day}_{index}"

    # update location data csv
    df_flowdeckdata_updated.to_csv(os.path.join('static', f'rawloc_{id}.csv'), index=False)
    
    # create csv containing image metadata
    df_imagedata.to_csv(os.path.join('static', f'imagedata_{id}.csv'), index=False)
    
    # extract zip image contents
    with zipfile.ZipFile(myzip, 'r') as file:
        os.mkdir(os.path.join(BASE_DIR,'static', f"images_{id}"))
        file.extractall(os.path.join(BASE_DIR,'static', f"images_{id}"))
    
    # clean uploads folder
    #os.remove(myzip)
    #os.remove(mycsv)

    return id

@measure_time
def YawTransform(id):
    
    # load flowdeck data
    df_flowdeckdata = pd.read_csv(os.path.join(BASE_DIR,'static', f'rawloc_{id}.csv'))
    x_coords = df_flowdeckdata['X']
    y_coords = df_flowdeckdata['Y']
    z_coords = df_flowdeckdata['Z']
    yaws = df_flowdeckdata['Yaw']
    timestamps = df_flowdeckdata['Timestamp']

    # for runs with no turns
    
    no_turn = True
    for yaw in yaws:
        if abs(yaw) > 30:
            no_turn = False
            break
    if no_turn:
        df_yawtrans = pd.DataFrame({
            'X': x_coords,
            'Y': y_coords,
            'Z': z_coords,
            'Yaw': yaws,
            'Timestamp': timestamps
        })
        df_yawtrans.to_csv(os.path.join('static', f'yawtrans_{id}.csv'), index=False)
        return

    # smoothen data
    x_coords = savgol_filter(x_coords, window_length=11, polyorder=10)
    y_coords = savgol_filter(y_coords, window_length=11, polyorder=10)
    z_coords = savgol_filter(z_coords, window_length=11, polyorder=10)
    yaws = savgol_filter(yaws, window_length=11, polyorder=10)

    # detect positive/negative peak yaw values and indices
    yaw_absolute = [abs(yaw) for yaw in yaws]
    yaw_diff = np.diff(yaw_absolute)
    yaw_diff = np.convolve(yaw_diff / np.max(yaw_diff), np.ones(3)/3, mode='valid')
    yaw_peaks_indices = np.where(yaw_diff > THRESHOLD_YAW)[0]

    # clean the peak indices
    temp = []
    for i, val in enumerate(np.diff(yaw_peaks_indices)):
        if abs(val) > 1:
            temp.append(yaw_peaks_indices[i-1])
    temp.append(yaw_peaks_indices[-1])
    yaw_peaks_indices = np.array(temp) + 5

    # retain yaw values at peaks
    yaw_retained = [0] * len(yaws)
    for i in range(1, len(yaws)):
        if i in yaw_peaks_indices:
            yaw_retained[i] = yaw_retained[i-1] + yaws[i]
        else:
            yaw_retained[i] = yaw_retained[i-1]

    # convert yaw in degrees to radians
    yaws_rad = np.radians(yaw_retained)

    # initializations for transformation
    prevx, prevy, vx, vy = 0, 0, 0, 0
    x_coords_transformed, y_coords_transformed  = [], []

    # rotation of coordinates
    for x, y, yaw in zip(x_coords, y_coords, yaws_rad):
        
        # calculate the new values of x and y coordinates
        x_new = round(vx + (x-prevx) * np.cos(yaw) - (y-prevy) * np.sin(yaw), 3)
        y_new = round(vy + (x-prevx) * np.sin(yaw) + (y-prevy) * np.cos(yaw), 3)
        
        # set the vertex and shifting factors for the next iteration
        prevx, prevy = x, y
        vx, vy = x_new, y_new
        
        # record the results
        x_coords_transformed.append(x_new)
        y_coords_transformed.append(y_new)
    
    # write the rotated coordinates and
    # retained yaw values to new csv
    df_yawtrans = pd.DataFrame({
        'X': x_coords_transformed,
        'Y': y_coords_transformed,
        'Z': z_coords,
        'Yaw': yaw_retained,
        'Timestamp': timestamps
    })
    df_yawtrans.to_csv(os.path.join('static', f'yawtrans_{id}.csv'), index=False)
    
    # visualization
    if SHOW_VISUALIZATION: 
        plt.plot(yaws, '-', label='Data')
        plt.plot(yaw_retained, '-', label='Retained')
        plt.scatter(yaw_peaks_indices, [yaws[i] for i in yaw_peaks_indices], color='red', label='Peaks', zorder=5)
        plt.legend()
        plt.show()

        plt.plot(yaws / np.max(abs(yaws)), '-', label='Original')
        plt.plot(yaw_diff / np.max(abs(yaw_diff)), '-', label='Diff')
        plt.scatter(yaw_peaks_indices, [(yaw_diff / np.max(abs(yaw_diff)))[i] for i in yaw_peaks_indices], color='red', label='Peaks', zorder=5)
        plt.legend()
        plt.show()

        plt.plot(x_coords, y_coords, label='Original')
        plt.plot(x_coords_transformed, y_coords_transformed, label='Transformed')
        plt.legend()
        plt.grid('on')
        plt.show()

    return

@measure_time
def MapGenerate(id):

    def smooth_coordinates(x, y, window_size=11, poly_order=2):
        # Ensure the window size is an odd integer
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer")
        
        # Apply Savitzky-Golay filter to smooth both x and y coordinates
        x_smooth = savgol_filter(x, window_length=window_size, polyorder=poly_order)
        y_smooth = savgol_filter(y, window_length=window_size, polyorder=poly_order)
        
        return x_smooth, y_smooth
    
    def simplify_path(x, y, tolerance=1.0):
        coords = np.column_stack((x, y))
        simplified = simplify_coords(coords, tolerance)
        return simplified[:, 0], simplified[:, 1]
    
    def find_largest_quadrilateral(x_coords, y_coords):
        max_area = 0
        best_quad = None
        n = len(x_coords)

        # Generate all possible combinations of 4 points
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        indices = [i, j, k, l]
                        x = [x_coords[idx] for idx in indices]
                        y = [y_coords[idx] for idx in indices]

                        # Compute area using Shoelace formula
                        area = 0.5 * abs(
                            x[0] * y[1] + x[1] * y[2] + x[2] * y[3] + x[3] * y[0]
                            - (y[0] * x[1] + y[1] * x[2] + y[2] * x[3] + y[3] * x[0])
                        )

                        # Update max area and best set of points
                        if area > max_area:
                            max_area = area
                            best_quad = [(x[m], y[m]) for m in range(4)]

        return best_quad
    
    def calculate_inclination_angle(x_coords, y_coords):
        if len(x_coords) != 2 or len(y_coords) != 2:
            raise ValueError("Input lists must each contain exactly two elements.")
        
        x1, x2 = x_coords
        y1, y2 = y_coords
        
        delta_x = x2 - x1
        delta_y = y2 - y1
        
        angle = math.atan2(delta_y, delta_x)
        return angle
    
    def shift_to_origin(x_coords, y_coords):
        # Convert the x and y coordinates to numpy arrays for easier manipulation
        points = np.array([x_coords, y_coords]).T
        
        # Calculate the centroid (mean of x and y coordinates)
        centroid = np.mean(points, axis=0)
        
        # Shift the points by subtracting the centroid
        shifted_points = points - centroid
        
        # Extract the shifted x and y coordinates
        shifted_x = shifted_points[:, 0].tolist()
        shifted_y = shifted_points[:, 1].tolist()
        
        # Calculate the shift amounts
        x_shift = centroid[0]
        y_shift = centroid[1]
        
        return shifted_x, shifted_y, x_shift, y_shift
    
    # Function to shift another shape by the same amount
    def shift_shape_by_amount(x_coords, y_coords, x_shift, y_shift):
        shifted_x = [x - x_shift for x in x_coords]
        shifted_y = [y - y_shift for y in y_coords]
        return shifted_x, shifted_y
    
    # Function to rotate a shape by an angle in radians
    def rotate_shape(x_coords, y_coords, angle):
        # Convert the x and y coordinates to numpy arrays for easier manipulation
        points = np.array([x_coords, y_coords]).T
        
        # Create the rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                    [np.sin(angle), np.cos(angle)]])
        
        # Rotate the points by multiplying with the rotation matrix
        rotated_points = np.dot(points, rotation_matrix.T)
        
        # Extract the rotated x and y coordinates
        rotated_x = rotated_points[:, 0].tolist()
        rotated_y = rotated_points[:, 1].tolist()
        
        return rotated_x, rotated_y
    
    def get_room_estimate(x_coords, y_coords):
        L = min(x_coords)
        R = max(x_coords)
        T = max(y_coords)
        B = min(y_coords)
        x_estimate = [L, L, R, R]
        y_estimate = [B, T, T, B]
        return x_estimate, y_estimate
    
    def shift_room_to_origin(x_coords, y_coords):
        # Convert to numpy arrays
        x = np.array(x_coords)
        y = np.array(y_coords)

        # Find the lower-leftmost point
        x_shift = min(x)
        y_shift = min(y)

        # Shift all points
        x_new = x - x_shift
        y_new = y - y_shift

        return x_new.tolist(), y_new.tolist(), x_shift, y_shift

    # load yawtrans data
    df_flowdeckdata = pd.read_csv(os.path.join(BASE_DIR,'static', f'yawtrans_{id}.csv'))
    x_coords = df_flowdeckdata['X']
    y_coords = df_flowdeckdata['Y']

    # smoothen data
    x_smooth, y_smooth = smooth_coordinates(x_coords, y_coords, window_size=5, poly_order=2)

    # simplify flight path
    x_simple, y_simple = simplify_path(x_smooth, y_smooth)

    # further simplify flight path
    best_points = find_largest_quadrilateral(x_simple, y_simple)
    best_points = find_largest_quadrilateral(x_simple, y_simple)
    x_flight = [p[0] for p in best_points]
    y_flight = [p[1] for p in best_points]
    
    # shift coordinates
    x_flight, y_flight, x_shift, y_shift = shift_to_origin(x_flight, y_flight)
    x_coords, y_coords  = shift_shape_by_amount(x_coords, y_coords, x_shift, y_shift)
    x_smooth, y_smooth  = shift_shape_by_amount(x_smooth, y_smooth, x_shift, y_shift)
    x_simple, y_simple  = shift_shape_by_amount(x_simple, y_simple, x_shift, y_shift)

    # rotate coordinates
    angle = calculate_inclination_angle(x_flight[:2], y_flight[:2])
    x_flight, y_flight = rotate_shape(x_flight, y_flight, -angle)
    x_coords, y_coords = rotate_shape(x_coords, y_coords, -angle)
    x_smooth, y_smooth = rotate_shape(x_smooth, y_smooth, -angle)
    x_simple, y_simple = rotate_shape(x_simple, y_simple, -angle)

    # form the room trace
    expansion_factor = DISTANCE_TO_WALL * np.sqrt(2)
    x_room = [x / expansion_factor for x in x_flight]
    y_room = [y / expansion_factor for y in y_flight]

    # form a perfected room estimate for web app presentation purposes
    x_estimate, y_estimate = get_room_estimate(x_room, y_room)

    # ensure room is of horizontal orientation for
    # web app viewing
    xlen = max(x_estimate) - min(x_estimate)
    ylen = max(y_estimate) - min(y_estimate)
    if ylen > xlen:
        x_flight, y_flight = rotate_shape(x_flight, y_flight, np.radians(90))
        x_coords, y_coords = rotate_shape(x_coords, y_coords, np.radians(90))
        x_smooth, y_smooth = rotate_shape(x_smooth, y_smooth, np.radians(90))
        x_simple, y_simple = rotate_shape(x_simple, y_simple, np.radians(90))
        x_room, y_room = rotate_shape(x_room, y_room, np.radians(90))
        x_estimate, y_estimate = rotate_shape(x_estimate, y_estimate, np.radians(90))

    # update global variable for flight path origin coordinates
    global x_flight_path_origin, y_flight_path_origin
    x_flight_path_origin = x_coords[0]
    y_flight_path_origin = y_coords[0]

    # update yawtrans data
    df_flowdeckdata['X'] = x_coords
    df_flowdeckdata['Y'] = y_coords

    # close the shapes for presentation purposes
    x_flight.append(x_flight[0])
    y_flight.append(y_flight[0])
    x_room.append(x_room[0])
    y_room.append(y_room[0])
    x_estimate.append(x_estimate[0])
    y_estimate.append(y_estimate[0])

    # shift lowerleftmost coordinate of room to origin
    # for neat web app presentation
    x_room_origin, y_room_origin, x_shift, y_shift = shift_room_to_origin(x_estimate, y_estimate)
    x_simple, y_simple  = shift_shape_by_amount(x_simple, y_simple, x_shift, y_shift)

    # update survey data csv
    df_flowdeckdata.to_csv(os.path.join('static', f'yawtrans_{id}.csv'), index=False)

    # visualization
    if SHOW_VISUALIZATION:
        plt.plot(x_coords, y_coords, '-', label="Raw")
        #plt.plot(x_smooth, y_smooth, '-', label="Smooth")
        #plt.plot(x_simple, y_simple, '-', label="Simple")
        plt.plot(x_flight, y_flight, '-', label="Flight Path, simplified")
        #plt.plot(x_room, y_room, '-', label="Room")
        plt.plot(x_estimate, y_estimate, '-', label="Room")
        plt.grid('on')
        plt.axis('square')
        plt.legend()
        plt.title('Map Generation')
        plt.show()

        plt.plot(x_room_origin, y_room_origin)
        plt.plot(x_simple, y_simple, '--', color='Blue')
        plt.axis('on')
        plt.grid('on')
        plt.fill(x_room_origin, y_room_origin, color='skyblue', edgecolor='black')
        plt.scatter(x_flight_path_origin - x_shift, y_flight_path_origin - y_shift, color="Blue", label="Origin")
        plt.text(x_flight_path_origin - x_shift, y_flight_path_origin - 0.3 - y_shift, "Flight Origin", ha='center', fontsize=8)
        plt.show()

    else:
        # Use the 'Agg' backend for rendering
        import matplotlib
        matplotlib.use('Agg')

        # Create map file
        mapfilename = os.path.join('static', f'map_{id}.png')

        plt.plot(x_room_origin, y_room_origin)
        plt.plot(x_simple, y_simple, '--', color='Blue')

        plt.axis('on')
        plt.grid('on')

        # Fill room shape
        plt.fill(x_room_origin, y_room_origin, color='skyblue', edgecolor='black')

        # Scatter flight origin
        plt.scatter(x_flight_path_origin - x_shift, y_flight_path_origin - y_shift, color="Blue", label="Origin")
        plt.text(x_flight_path_origin - x_shift, y_flight_path_origin - 0.3 - y_shift, "Flight Origin",
                ha='center', fontsize=8)

        # Get current axis
        ax = plt.gca()

        # Move tick labels inside
        ax.tick_params(axis='x', direction='in', pad=-15)
        ax.tick_params(axis='y', direction='in', pad=-25)

        # Ensure only whole number ticks
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Remove tick labels for (0,0) while keeping other ticks
        ax.set_xticklabels(['' if tick == 0 else str(int(tick)) for tick in ax.get_xticks()])
        ax.set_yticklabels(['' if tick == 0 else str(int(tick)) for tick in ax.get_yticks()])

        # Save figure
        plt.savefig(mapfilename, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()

    return x_flight, y_flight, x_estimate, y_estimate

@measure_time
def TimeMatch(id):

    # load yawtrans location data
    df_yawtransdata = pd.read_csv(os.path.join(BASE_DIR,'static',  f'yawtrans_{id}.csv'))

    # load image metadata 
    df_imagedata = pd.read_csv(os.path.join(BASE_DIR,'static', f'imagedata_{id}.csv'))

    one_time = []
    unique_times = df_yawtransdata['Timestamp'].unique().tolist()

    for each in unique_times:
        # filter the df by each unique timestamp
        filtered_time = df_yawtransdata[df_yawtransdata['Timestamp'] == each].copy()
        
        # calculate the mean for the filtered df
        mean_values = filtered_time.drop(columns='Timestamp').mean()
        
        # calculate the distance for the filtered DataFrame and set it using .loc
        filtered_time.loc[:, 'Distance_From_Mean'] = np.sqrt(((filtered_time.drop(columns='Timestamp') - mean_values) ** 2).sum(axis=1))
        
        # append the row with the minimum distance to the list
        closest_sample = filtered_time.loc[filtered_time['Distance_From_Mean'].idxmin()].drop(labels='Distance_From_Mean')
        one_time.append(closest_sample.tolist())

    # convert the list into a df
    df_yawtransdata = pd.DataFrame(one_time, columns=df_yawtransdata.columns)

    # match image time with raw location data
    df_surveydata = pd.merge(df_yawtransdata, df_imagedata[['ImagePath', 'Timestamp']], on='Timestamp')

    # structure dataframe to drop or include necessary columns
    df_surveydata = df_surveydata.drop('Timestamp', axis=1)
    df_surveydata['Classification'] = 'unclassified'
    df_surveydata['GridLabel'] = 0
    df_surveydata['BooleanShow'] = 1
    df_surveydata['Notes'] = 'This is a note.'

    # update the image file paths
    imagepaths = df_surveydata['ImagePath'].tolist()
    for i, imagepath in enumerate(imagepaths):
        imagepaths[i] = os.path.join(f"images_{id}", imagepath).replace('\\', '/')
    df_surveydata['ImagePath'] = imagepaths

    # save the time matched dataframe to csv with proper filename
    df_surveydata.to_csv(os.path.join('static', f"survey_{id}.csv"), sep='|', index=False)

    # remove unmatched images from the folder
    folder_path = os.path.join(BASE_DIR,'static', f"images_{id}")
    folder_files = os.listdir(folder_path)
    for file_name in folder_files:
        if os.path.join(f"images_{id}", file_name).replace('\\', '/') not in imagepaths:
            file_path = os.path.join(folder_path, file_name)
            # Ensure it's a file before deleting
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    return

@measure_time
def GridAssign(id, x_corners, y_corners, x_room, y_room):
    
    def calculate_distance(x1, y1, x2, y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def calculate_centroid(x_coords, y_coords):

        if len(x_coords) != len(y_coords):
            raise ValueError("The number of x-coordinates and y-coordinates must be the same.")
        
        # Calculate the averages
        x_centroid = sum(x_coords) / len(x_coords)
        y_centroid = sum(y_coords) / len(y_coords)

        return x_centroid, y_centroid

    def translate_coordinates(x_coords, y_coords, x_shift, y_shift):
        xc, yc = calculate_centroid(x_coords, y_coords)
        return [x + x_shift for x in x_coords], [y + y_shift for y in y_coords]

    def get_nearest_grid_label(x_query, y_query, unique_x, unique_y, grid_labels):
        # Find nearest x-coordinate
        x_idx = np.searchsorted(unique_x, x_query, side="left")
        if x_idx == 0:
            nearest_x = unique_x[0]
        elif x_idx == len(unique_x):
            nearest_x = unique_x[-1]
        else:
            left_x, right_x = unique_x[x_idx - 1], unique_x[x_idx]
            nearest_x = left_x if abs(left_x - x_query) < abs(right_x - x_query) else right_x

        # Find nearest y-coordinate
        y_idx = np.searchsorted(unique_y, y_query, side="left")
        if y_idx == 0:
            nearest_y = unique_y[0]
        elif y_idx == len(unique_y):
            nearest_y = unique_y[-1]
        else:
            left_y, right_y = unique_y[y_idx - 1], unique_y[y_idx]
            nearest_y = left_y if abs(left_y - y_query) < abs(right_y - y_query) else right_y

        # Retrieve label
        return grid_labels.get((nearest_x, nearest_y), None)

    def rotate_points(x_coords, y_coords, angle, center):
        """Rotate points by a given angle around a center point."""
        # Combine x and y coordinates into a 2D array
        points = np.array(list(zip(x_coords, y_coords)))

        # Rotation matrix for 2D
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        
        # Center the points around the rotation center
        centered_points = points - center
        rotated_points = np.dot(centered_points, rotation_matrix.T)
        
        # Translate the rotated points back to the original position
        rotated_points += center

        # Extract the rotated x and y coordinates as separate lists
        rotated_x = rotated_points[:, 0].tolist()
        rotated_y = rotated_points[:, 1].tolist()

        return rotated_x, rotated_y

    # load detected crack data
    df_surveydata = pd.read_csv(os.path.join(BASE_DIR,'static', f"survey_{id}.csv"), sep='|')
    x_crack = df_surveydata['X'].tolist()
    y_crack = df_surveydata['Y'].tolist()

    shorter = 100

    factor = 0
    xmax = max(x_corners) + factor
    xmin = min(x_corners) - factor
    ymax = max(y_corners) + factor*2
    ymin = min(y_corners) - factor*2
    
    c = int(xmax - xmin) + 1
    r = int(ymax - ymin) + 1
    rows, cols = 0, 0

    if c > r:
        rows = shorter
        cols = int(shorter * (c/r))
    elif c < r:
        rows = int(shorter * (r/c))
        cols = shorter
    else:
        rows, cols = shorter, shorter

    # generate cluster coordinates
    x_cluster_points = np.linspace(xmin, xmax, cols).tolist()
    y_cluster_points = np.linspace(ymin, ymax, rows).tolist()
    y_cluster_points_rev = list(reversed(y_cluster_points))
    x_flight_grid, y_flight_grid = [], []
    for y in y_cluster_points_rev:
        for x in x_cluster_points:
            if y in [ymin, ymax] or x in [xmin, xmax]:
                x_flight_grid.append(x)
                y_flight_grid.append(y)

    # fit cracks to flight grid
    x_fit_flight, y_fit_flight = [], []
    for x, y in zip(x_crack, y_crack):
        least_distance = float('inf')
        xpos, ypos = 0, 0
        for x_grid, y_grid in zip(x_flight_grid, y_flight_grid):
            distance = calculate_distance(x, y, x_grid, y_grid)
            if distance < least_distance:
                least_distance = distance
                xpos, ypos = x_grid, y_grid
        x_fit_flight.append(xpos)
        y_fit_flight.append(ypos)
    
    expansion_factor = DISTANCE_TO_WALL * np.sqrt(2)
    x_crack_room = [x / expansion_factor for x in x_fit_flight]
    y_crack_room = [y / expansion_factor for y in y_fit_flight]

    xmax = max(x_room)
    xmin = min(x_room)
    ymax = max(y_room)
    ymin = min(y_room)

    # generate cluster coordinates
    x_room_grid, y_room_grid = [], []
    x_cluster_points = np.linspace(xmin, xmax, cols).tolist()
    y_cluster_points = np.linspace(ymin, ymax, rows).tolist()
    y_cluster_points_rev = list(reversed(y_cluster_points))
    x_flight_grid, y_flight_grid = [], []
    for y in y_cluster_points_rev:
        for x in x_cluster_points:
            if y in [ymin, ymax] or x in [xmin, xmax]:
                x_room_grid.append(x)
                y_room_grid.append(y)

    # fit cracks to room grid
    x_fit_room, y_fit_room = [], []
    for x, y in zip(x_crack_room, y_crack_room):
        least_distance = float('inf')
        xpos, ypos = 0, 0
        for x_grid, y_grid in zip(x_room_grid, y_room_grid):
            distance = calculate_distance(x, y, x_grid, y_grid)
            if distance < least_distance:
                least_distance = distance
                xpos, ypos = x_grid, y_grid
        x_fit_room.append(xpos)
        y_fit_room.append(ypos)

    # generate meshgrid
    # Create meshgrid for all (x, y) pairs
    xmax = max(x_room_grid)
    xmin = min(x_room_grid)
    ymax = max(y_room_grid)
    ymin = min(y_room_grid)

    x_coords = np.linspace(xmin, xmax, cols)
    y_coords = np.linspace(ymin, ymax, rows)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Flatten meshgrid to match coordinate lists
    x_flat = X.ravel()
    y_flat = Y.ravel()

    # Get sorted unique values
    unique_x = np.sort(np.unique(x_flat))
    unique_y = np.sort(np.unique(y_flat))

    # Create mapping for indices
    x_indices = {val: i for i, val in enumerate(unique_x)}
    y_indices = {val: i for i, val in enumerate(unique_y)}

    # Assign grid numbers row-wise (top to bottom, left to right)
    grid_labels = { (x, y): (rows - 1 - y_indices[y]) * cols + x_indices[x] + 1
                for x, y in zip(x_flat, y_flat) }

    # Assign labels to input coordinates
    assigned_labels = np.array([grid_labels[(x, y)] for x, y in zip(x_flat, y_flat)])

    # Reshape into grid form
    grid_matrix = assigned_labels.reshape(rows, cols)

    # Example query points (x, y) and grid setup
    x_query = x_fit_room
    y_query = y_fit_room

    # Combine query points into a 2D array
    query_points = np.vstack([x_query, y_query]).T

    # Use DBSCAN to cluster points with epsilon=0.2
    dbscan = DBSCAN(eps=0.5, min_samples=1)  # min_samples=1 means even single points are considered a cluster
    labels = dbscan.fit_predict(query_points)

    # Assign grid labels to the query points based on cluster centers
    cluster_labels = []
    for cluster_id in np.unique(labels):
        # Get the center of the cluster (mean of points in that cluster)
        cluster_points = query_points[labels == cluster_id]
        center_x = np.mean(cluster_points[:, 0])
        center_y = np.mean(cluster_points[:, 1])
        
        # Find nearest grid label for the cluster center
        nearest_label = get_nearest_grid_label(center_x, center_y, unique_x, unique_y, grid_labels)
        
        # Assign this grid label to all points in the cluster
        cluster_labels.append((cluster_id, nearest_label))

    # Output the cluster labels
    clustered_points = [(query_points[i], labels[i]+1, cluster_labels[labels[i]][1]-1) for i in range(len(query_points))]

    df = pd.DataFrame(clustered_points, columns=['Query Points', 'Position', 'GridLabel'])

    df_surveydata['GridLabel'] = df['GridLabel']
    df_surveydata['Position'] = df['Position']
    df_surveydata['X'] = x_fit_room
    df_surveydata['Y'] = y_fit_room

    # update survey data csv
    df_surveydata.to_csv(os.path.join('static', f"survey_{id}.csv"), sep='|', index=False)

    # visualization
    if SHOW_VISUALIZATION:
        plt.plot(x_flight_grid, y_flight_grid, '.', label='Flight Grid')
        plt.plot(x_room_grid, y_room_grid, '.', label='Room')
        plt.plot(x_crack, y_crack, 'x', label='Cracks')
        plt.plot(x_fit_flight, y_fit_flight, 'x', label='Cracks, fit to Flight Grid')
        plt.plot(x_fit_room, y_fit_room, 'x', label='Cracks, fit to Room Trace')
        plt.grid('on')
        plt.legend()
        plt.show()

    return rows, cols

@measure_time
def CrackClassifier(id):

    # get survey data
    df_surveydata = pd.read_csv(os.path.join(BASE_DIR,'static', f"survey_{id}.csv"), sep='|')

    # classify the images

    # load the model
    model = YOLO("assets/best.pt")

    # declare blank arrays
    image_paths = []
    classes = []

    for file in df_surveydata['ImagePath'].tolist():
        # Check if file is an absolute path (starts with '/home/superlinkfour/crackfinder_webapp')
        if file.startswith(BASE_DIR):  # BASE_DIR is '/home/superlinkfour/crackfinder_webapp'
            # Extract the relative path from the BASE_DIR
            file = file[len(BASE_DIR):]  # Remove BASE_DIR prefix from file path
    
        # Ensure 'static/' is part of the image path
        if not file.startswith('static/'):
            file = 'static/' + file  # Prepend static/ to the path if missing

        # Construct the full image path
        img_path = os.path.join(BASE_DIR, file)  # Full path to image

        image_paths.append(img_path)

    # Debugging: Check the final paths
    for img_path in image_paths:
        print(f"🚀 DEBUG: Final image path used: {img_path}")

    # prep the images
    images = [cv2.cvtColor(cv2.imread(os.path.join(BASE_DIR, 'static', img_path)), cv2.COLOR_BGR2RGB) for img_path in image_paths]
    if not images:
        raise ValueError("❌ ERROR: No images to process. Ensure image paths are valid.")

    # make predictions on the images
    results = model(images)

    # process results for each image
    classes = []
    for result in results:
        # this contains the probability for each class
        class_probs = result.probs.cpu().numpy().data

        # predicted class is the class with highest probability
        predicted_class_index = np.argmax(class_probs)
        predicted_class_name = result.names[predicted_class_index]

        """
        remove if-statement below if we want to still
        see negative images for testing purposes
        if predicted_class_name == 'negative':
            continue
        """
        # record the predicted class name
        classes.append(predicted_class_name)

    # add classifications columns to dataframe
    df_surveydata['Classification'] = classes

    # update csv
    df_surveydata.to_csv(os.path.join(BASE_DIR,'static', f"survey_{id}.csv"), sep='|', index=False)

    return

@measure_time
def DataConsolidation(id, rows, cols):

    # load the contents of sessions.csv
    df_sessions = pd.read_csv('sessions.csv', sep='|')
    ids = df_sessions['ID'].tolist()
    dates = df_sessions['DateYMD'].tolist()
    venues = df_sessions['Venue'].tolist()
    crackcounts = df_sessions['CrackCount'].tolist()
    notes = df_sessions['Notes'].tolist()
    csvfilenames = df_sessions['CsvFilename'].tolist()
    mapfilenames = df_sessions['MapFilename'].tolist()
    rows_entries = df_sessions['Rows'].tolist()
    cols_entries = df_sessions['Cols'].tolist()

    # gather data to append:

    # survey session id
    ids.append(id)

    # survey date
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})', id)
    year, month, day = match.groups()
    dates.append(f"{year}-{month}-{day}")

    # map image filename
    filename = os.path.join('static', f'map_{id}.png')
    mapfilenames.append(filename)

    # survey csv filename
    filename = os.path.join('static', f'survey_{id}.csv')
    csvfilenames.append(filename)

    # count of cracks photo-taken
    df = pd.read_csv(filename, sep='|')
    crackcounts.append(df[df["Classification"] != "negative"].shape[0])

    # number of rows and columns
    rows_entries.append(rows)
    cols_entries.append(cols)

    # placeholder venues for others
    venues.append('None added.')
    notes.append('None added.')

    # update the dataframe
    new_df_sessions = pd.DataFrame()
    new_df_sessions['ID'] = ids
    new_df_sessions['DateYMD'] = dates
    new_df_sessions['Venue'] = venues
    new_df_sessions['CrackCount'] = crackcounts
    new_df_sessions['Notes'] = notes
    new_df_sessions['CsvFilename'] = csvfilenames
    new_df_sessions['MapFilename'] = mapfilenames
    new_df_sessions['Rows'] = rows_entries
    new_df_sessions['Cols'] = cols_entries

    # update the csv file
    new_df_sessions.to_csv('sessions.csv', sep='|', index=False)

    # assign flag colors
    df = pd.read_csv(os.path.join('static', f"survey_{id}.csv"), sep='|')

    # Priority order mapping
    priority = {'horizontal': 1, 'diagonal': 2, 'vertical': 3, 'negative': 4}
    color_mapping = {1: '#D32F2F', 2: '#F57C00', 3: '#F57C00', 4: '#388E3C'}

    # Function to get the flag color for each unique position
    def assign_flag_color(classifications):
        min_priority = min(priority.get(c, 4) for c in classifications)  # Default to lowest priority
        return color_mapping[min_priority]

    # Group by Position and assign flag color based on highest priority classification
    df['FlagColor'] = df.groupby('Position')['Classification'].transform(assign_flag_color)

    # Update the csv
    df.to_csv(os.path.join('static', f"survey_{id}.csv"), sep='|', index=False)

    return

@measure_time
def CrackGPT():
    id = DataPreparation()
    YawTransform(id)
    x_flight, y_flight, x_estimate, y_estimate = MapGenerate(id)
    TimeMatch(id)
    rows, cols = GridAssign(id, x_flight, y_flight, x_estimate, y_estimate)
    CrackClassifier(id)
    DataConsolidation(id, rows=rows, cols=cols)
    return
