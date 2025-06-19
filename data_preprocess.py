def find_min_numbered_folder(base_path):
    if not os.path.isdir(base_path):
        return None
    numbered_folders_info = []
    for item in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, item)) and item.isdigit():
            numbered_folders_info.append((int(item), item)) 
    if numbered_folders_info:
        numbered_folders_info.sort(key=lambda x: x[0])
        return numbered_folders_info[0][1]
    return None

#加载xray
def load_xray_images_for_patient(xray_patient_folder_path, knee_side_folders=["1_JPG", "2_JPG"]):
    loaded_images = []
    x_folder_path = os.path.join(xray_patient_folder_path, 'X')
    if not os.path.isdir(x_folder_path):
        return [None] * len(knee_side_folders)
    min_num_folder = find_min_numbered_folder(x_folder_path)
    if min_num_folder is None:
        return [None] * len(knee_side_folders)
    for knee_side_folder in knee_side_folders:
        target_folder = os.path.join(x_folder_path, min_num_folder, knee_side_folder)
        if not os.path.isdir(target_folder):
            loaded_images.append(None) 
            continue
        
        found_image = False
        for filename in sorted(os.listdir(target_folder)):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                image_path = os.path.join(target_folder, filename)
                try:
                    loaded_images.append(Image.open(image_path).convert('RGB'))
                    found_image = True
                    break 
                except Exception as e:
                    print(f"Error opening X-ray image {image_path}: {e}")
                    loaded_images.append(None)
                    found_image = True # Mark as attempted, but failed
                    break
        if not found_image:

            loaded_images.append(None) 
    return loaded_images

#加载mri
def load_and_preprocess_dicom_slice(dicom_filepath):

    try:
        dicom_data = pydicom.dcmread(dicom_filepath)
        if 'RescaleSlope' in dicom_data and 'RescaleIntercept' in dicom_data:
            slope = dicom_data.RescaleSlope
            intercept = dicom_data.RescaleIntercept
            image_array = dicom_data.pixel_array * slope + intercept
        else:
            image_array = dicom_data.pixel_array
        image_array = image_array.astype(np.float32)
        if image_array.max() == image_array.min():
            image_array = np.zeros_like(image_array) 
        else:
            image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255.0
        if image_array.ndim == 3:
            # 取中间切片。
            image_array = image_array[:, :, image_array.shape[2] // 2] 
            
        return Image.fromarray(image_array.astype(np.uint8)).convert('RGB')
    except Exception as e:
        print(f"Error loading and processing DICOM slice {dicom_filepath}: {e}")
        return None

def get_mri_representative_slices_for_patient(mri_patient_folder_path, sequence_types=["PDWI-SAG"], slice_selection="middle"):
    loaded_images = []
    mri_study_folder = os.path.join(mri_patient_folder_path, 'MRI')
    if not os.path.isdir(mri_study_folder):
        return [None] * len(sequence_types)

    min_num_folder = find_min_numbered_folder(mri_study_folder)
    if min_num_folder is None:
        return [None] * len(sequence_types)

    for sequence_type in sequence_types:
        sequence_dir = os.path.join(mri_study_folder, min_num_folder, sequence_type)
        if not os.path.isdir(sequence_dir):
            loaded_images.append(None)
            continue
        dicom_filepaths = []
        for filename in os.listdir(sequence_dir):
            filepath = os.path.join(sequence_dir, filename)
            if os.path.isfile(filepath) and (filename.lower().endswith('.dcm') or '.' not in filename):
                dicom_filepaths.append(filepath)
        if not dicom_filepaths:
            loaded_images.append(None)
            continue

        dicom_info = []
        for fp in dicom_filepaths:
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True) 
                instance_number = getattr(ds, 'InstanceNumber', 0) 
                slice_location = getattr(ds, 'SliceLocation', 0.0) 
                dicom_info.append({'path': fp, 'instance_number': int(instance_number), 'slice_location': float(slice_location)})
            except Exception as e:
                continue

        if not dicom_info:
            loaded_images.append(None)
            continue
        try:
            dicom_info.sort(key=lambda x: x['instance_number'])
        except Exception: 
            try:
                dicom_info.sort(key=lambda x: x['slice_location'])
            except Exception:
                dicom_info.sort(key=lambda x: x['path'])
        selected_image = None
        if slice_selection == "middle":
            selected_filepath = dicom_info[len(dicom_info) // 2]['path']
            selected_image = load_and_preprocess_dicom_slice(selected_filepath)
        elif slice_selection == "first":
            selected_filepath = dicom_info[0]['path']
            selected_image = load_and_preprocess_dicom_slice(selected_filepath)
        elif slice_selection == "last":
            selected_filepath = dicom_info[-1]['path']
            selected_image = load_and_preprocess_dicom_slice(selected_filepath)
        else:
            print(f"Invalid slice_selection: {slice_selection}. Returning None for this sequence.")
        
        loaded_images.append(selected_image)

    return loaded_images


class KneeMultiModalDataset(Dataset):
    def __init__(self, df, xray_base_dir, mri_base_dir, mri_sequence_types=["PDWI-SAG"],
                 xray_knee_sides=["1_JPG"], transform=None):
        self.df = df
        self.xray_base_dir = xray_base_dir
        self.mri_base_dir = mri_base_dir
        self.mri_sequence_types = mri_sequence_types
        self.xray_knee_sides = xray_knee_sides
        self.transform = transform

        self.patient_folder_map = self._create_patient_folder_map()
        self.available_patients_df = self.df[self.df['患者姓名'].isin(self.patient_folder_map.keys())].reset_index(drop=True)
        print(f"Found {len(self.available_patients_df)} patients with complete X-ray and MRI data out of {len(self.df)} in CSV.")
        if len(self.available_patients_df) == 0:
            raise ValueError("No patients found with complete X-ray and MRI data based on provided paths and CSV. Please check folder names and CSV entries.")


    def _create_patient_folder_map(self):
        patient_map = {}
        xray_patient_folder_paths = {}
        if os.path.isdir(self.xray_base_dir):
            for folder_name in os.listdir(self.xray_base_dir):
                full_path = os.path.join(self.xray_base_dir, folder_name)
                if os.path.isdir(full_path):
                    parts = folder_name.split('.', 1) 
                    if len(parts) > 1:
                        patient_name = parts[1].strip()
                        xray_patient_folder_paths[patient_name] = full_path

        mri_patient_folder_paths = {}
        if os.path.isdir(self.mri_base_dir):
            for folder_name in os.listdir(self.mri_base_dir):
                full_path = os.path.join(self.mri_base_dir, folder_name)
                if os.path.isdir(full_path):
                    parts = folder_name.split('.', 1)
                    if len(parts) > 1:
                        patient_name = parts[1].strip()
                        mri_patient_folder_paths[patient_name] = full_path
        
        for patient_name_from_csv in self.df['患者姓名'].unique():
            if patient_name_from_csv in xray_patient_folder_paths and \
               patient_name_from_csv in mri_patient_folder_paths:
                patient_map[patient_name_from_csv] = {
                    'xray_folder': xray_patient_folder_paths[patient_name_from_csv],
                    'mri_folder': mri_patient_folder_paths[patient_name_from_csv]
                }
        return patient_map

    def __len__(self):
        return len(self.available_patients_df)

    def __getitem__(self, idx):
        patient_name = self.available_patients_df.iloc[idx]['患者姓名']
        kl_score = self.available_patients_df.iloc[idx]['KL得分']

        paths = self.patient_folder_map.get(patient_name)
        if paths is None:
            raise RuntimeError(f"Internal error: Paths for patient {patient_name} not found in map.")
        xray_images = load_xray_images_for_patient(paths['xray_folder'], knee_side_folders=self.xray_knee_sides)
        processed_xray_images = []
        for img in xray_images:
            if img is None:
                # print(f"Warning: Missing X-ray image for patient '{patient_name}' in one of the specified sides. Using black placeholder.")
                processed_xray_images.append(self.transform(Image.new('RGB', (224, 224), color='black')))
            else:
                processed_xray_images.append(self.transform(img))
        mri_images = get_mri_representative_slices_for_patient(paths['mri_folder'], sequence_types=self.mri_sequence_types, slice_selection="middle")
        processed_mri_images = []
        for img in mri_images:
            if img is None:
                # print(f"Warning: Missing MRI image for patient '{patient_name}' in one of the specified sequences. Using black placeholder.")
                processed_mri_images.append(self.transform(Image.new('RGB', (224, 224), color='black')))
            else:
                processed_mri_images.append(self.transform(img))
        return processed_xray_images, processed_mri_images, torch.tensor(kl_score, dtype=torch.long)