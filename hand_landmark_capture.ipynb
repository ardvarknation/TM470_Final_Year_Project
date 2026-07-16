# Initialise MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 

# Directories containing data with labelled folders of images containing BSL Alphanumeric signs.
DATA_DIRS = ['/content/.../BSL_Alphabet', '/content/.../BSL_Numbers']

# Directory to store the output CSV file.
OUTPUT_CSV = '/content/.../labelled_bsl_data.csv'

# Prepare a CSV file to write to.
with open(OUTPUT_CSV, node='w', newline='') as file:
  writer = csv.writer(file)

  # Write the header row.
  header = []
  # Each hand has 21 landmarks (x, y, z coordinates).
  for hand_idx in range(2):    # Detects up to 2 hands in image.
    for lm_idx in range(21):
      header.extend([f'hand{hand_idx+1}_lm{lm_idx}_y', f'hand{hand_idx+1}_lm{lm_idx}_z'])
  header.append('label')
  writer.writerow(header)

  # Process each directory in datasets.
  for data_dir in DATA_DIRS:
    # Process each labelled folder and parse label.
    for label in os.listdir(data_dir):
      label_path = os.path.join(data_dir, label)
      if not os.path.isdir(label_path):
        continue

      # Process each image in each folder.
      for image_name in os.listdir(label_path):
        image_path = os.path.join(label_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
          continue

        # Convert image to format required for MediaPipe.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Initialise a list to hold landmark data for up to two hands per image.
        all_hand_landmarks_data = []
        for _ in range(2):      # Initialise with None or empty list for two hands.
          all_hand_landmarks_data.append(None)

        if results.multi_hand_landmarks:
          for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx < 2:    # Process up to two hands per image.
              landmarks = hand_landmarks.landmark
              # Normalize by translating wrist (landmark 0) to origin.
              origin = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
              normalized_landmarks = []
              for lm in landmarks:
                normalized = np.array([lm.x, lm.y, lm.z])
                normalized_landmarks.extend(normalized.tolist())
              all_hand_landmarks_data[hand_idx] = normalized_landmarks

        # Prepare the row for CSV.
        row_data = []
        for hand_data in all_hand_landmarks_data:
          if hand_data is not None:
            row_data.extend(hand_data)
          else:
            # If a hand was not detected, append zeros for its landmarks.
            row_data.extend([0.0] * (21 * 3))  # 21 landmarks * 3 coordinates (x, y, z).

        # Add the label to the end of the row.
        row_data.append(label)

        # Write the row of data with corresponding label to CSV file as (features, label) pair.
        writer.writerow(row_data)
        
          
