import streamlit as st
import sys
import os

# Force reload of page if there are import issues
st.set_page_config(
    page_title="Microstructure Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try importing with fallbacks
try:
    import cv2
    import numpy as np
    import pandas as pd
    from skimage import filters, morphology, measure, segmentation
    from scipy import ndimage as ndi
    import matplotlib.pyplot as plt
    from PIL import Image
    import io
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("""
    Please make sure you have Python 3.11 or 3.12 installed.
    
    Create a file called `runtime.txt` with content: `python-3.11`
    Then redeploy your app.
    """)
    st.stop()

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Microstructure Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 1
if 'raw_image' not in st.session_state:
    st.session_state.raw_image = None
if 'preprocessed_image' not in st.session_state:
    st.session_state.preprocessed_image = None
if 'calibration_um_per_pixel' not in st.session_state:
    st.session_state.calibration_um_per_pixel = None
if 'material_type' not in st.session_state:
    st.session_state.material_type = "Dense ceramic"
if 'segmentation_results' not in st.session_state:
    st.session_state.segmentation_results = None
if 'calibration_line_left' not in st.session_state:
    st.session_state.calibration_line_left = 0.2
if 'calibration_line_right' not in st.session_state:
    st.session_state.calibration_line_right = 0.8
if 'calibration_known_um' not in st.session_state:
    st.session_state.calibration_known_um = 100

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_image(uploaded_file):
    """Load image from uploaded file"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def preprocess_image(image, clahe_clip=2.0, clahe_grid=(8,8), denoise=True):
    """Apply preprocessing: denoising, CLAHE, contrast enhancement"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Denoise if needed
    if denoise:
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)
    
    return enhanced

def adaptive_binarization(image, method='otsu', block_size=51, c=5):
    """Adaptive binarization"""
    if method == 'otsu':
        thresh = filters.threshold_otsu(image)
        binary = (image > thresh).astype(np.uint8)
    elif method == 'adaptive_gaussian':
        binary = cv2.adaptiveThreshold(
            image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c
        )
    else:  # local mean
        binary = cv2.adaptiveThreshold(
            image, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
    return binary

def segment_grains_watershed(binary_image, original_gray, min_distance=10):
    """Watershed segmentation with markers"""
    # Distance transform
    distance = ndi.distance_transform_edt(binary_image)
    
    # Find local maxima as markers
    local_max = morphology.local_maxima(distance, footprint=np.ones((min_distance, min_distance)))
    markers = measure.label(local_max)
    
    # Watershed
    segmented = segmentation.watershed(-distance, markers, mask=binary_image)
    
    return segmented

def segment_grains_ananyev(image_gray):
    """
    Simplified Ananyev et al. algorithm:
    1. Adaptive smoothing
    2. Gaussian blur
    3. Illumination equalization
    4. Adaptive binarization
    5. Morphological corrections
    """
    # Step 1: Adaptive smoothing
    smoothed = cv2.medianBlur(image_gray, 5)
    
    # Step 2: Gaussian blur
    blurred = cv2.GaussianBlur(smoothed, (15, 15), 3)
    
    # Step 3: Illumination equalization (division)
    # Avoid division by zero
    blurred = np.maximum(blurred, 1)
    equalized = (image_gray.astype(np.float32) / blurred.astype(np.float32)) * 255
    equalized = np.clip(equalized, 0, 255).astype(np.uint8)
    
    # Step 4: Adaptive binarization
    binary = cv2.adaptiveThreshold(
        equalized, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 3
    )
    
    # Step 5: Morphological cleaning
    kernel = np.ones((3,3), np.uint8)
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=50)
    binary = binary.astype(np.uint8)
    
    return binary

def extract_grain_properties(segmented_image, calibration_um_per_pixel):
    """Extract grain properties: area, perimeter, equivalent diameter"""
    props_list = []
    regions = measure.regionprops(segmented_image)
    
    for region in regions:
        if region.area < 50:  # skip too small
            continue
            
        area_um2 = region.area * (calibration_um_per_pixel ** 2)
        perimeter_um = region.perimeter * calibration_um_per_pixel
        equivalent_diameter_um = np.sqrt(4 * area_um2 / np.pi)
        
        props_list.append({
            'label': region.label,
            'area_pixels': region.area,
            'area_um2': area_um2,
            'perimeter_um': perimeter_um,
            'eq_diameter_um': equivalent_diameter_um,
            'centroid_x': region.centroid[1],
            'centroid_y': region.centroid[0]
        })
    
    return pd.DataFrame(props_list)

def calculate_porosity(binary_grains, binary_pores=None):
    """Calculate porosity percentage"""
    if binary_pores is None:
        # Assume pores are dark regions
        binary_pores = (binary_grains == 0).astype(np.uint8)
    
    total_pixels = binary_pores.size
    pore_pixels = np.sum(binary_pores == 1)
    porosity = (pore_pixels / total_pixels) * 100
    
    return porosity

def create_calibration_figure(image, left_pos, right_pos):
    """Create interactive figure for scale calibration"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    
    h, w = image.shape[:2]
    left_x = int(left_pos * w)
    right_x = int(right_pos * w)
    
    # Draw calibration lines
    ax.axvline(x=left_x, color='red', linewidth=2, linestyle='--', label='Left marker')
    ax.axvline(x=right_x, color='blue', linewidth=2, linestyle='--', label='Right marker')
    
    # Draw line between markers
    ax.plot([left_x, right_x], [h//2, h//2], 'g-', linewidth=2, label='Measured distance')
    
    ax.set_title('Click and drag sliders below to adjust markers')
    ax.axis('off')
    ax.legend(loc='upper right')
    
    return fig

# ============================================================================
# STAGE 1: IMAGE UPLOAD AND PREPROCESSING
# ============================================================================

st.title("🔬 Microstructure Analyzer for Ceramic & Metal Samples")
st.markdown("---")

if st.session_state.stage == 1:
    st.header("📤 Stage 1: Image Upload & Preprocessing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image (SEM, optical microscopy)",
            type=['png', 'jpg', 'jpeg', 'tif', 'bmp']
        )
        
        if uploaded_file is not None:
            raw_image = load_image(uploaded_file)
            st.session_state.raw_image = raw_image
            st.image(raw_image, caption="Original Image", use_container_width=True)
    
    with col2:
        if st.session_state.raw_image is not None:
            st.subheader("Preprocessing Settings")
            
            clahe_clip = st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, 0.5)
            denoise = st.checkbox("Apply denoising", value=True)
            
            if st.button("🔄 Apply Preprocessing", type="primary"):
                with st.spinner("Processing image..."):
                    processed = preprocess_image(
                        st.session_state.raw_image,
                        clahe_clip=clahe_clip,
                        denoise=denoise
                    )
                    st.session_state.preprocessed_image = processed
                st.success("Preprocessing complete!")
    
    if st.session_state.preprocessed_image is not None:
        st.image(
            st.session_state.preprocessed_image,
            caption="Preprocessed Image",
            use_container_width=True,
            clamp=True
        )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➡️ Next: Calibration", type="primary", use_container_width=True):
            if st.session_state.preprocessed_image is not None:
                st.session_state.stage = 2
                st.rerun()
            else:
                st.error("Please upload and preprocess an image first")

# ============================================================================
# STAGE 2: CALIBRATION & MATERIAL SELECTION
# ============================================================================

elif st.session_state.stage == 2:
    st.header("📏 Stage 2: Scale Calibration & Material Selection")
    
    if st.session_state.preprocessed_image is None:
        st.error("No preprocessed image found. Please go back to Stage 1.")
        if st.button("◀️ Back to Stage 1"):
            st.session_state.stage = 1
            st.rerun()
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Calibration")
            st.markdown("""
            **How to calibrate:**
            1. Move the left and right sliders to align markers with your scale bar
            2. Enter the known distance in micrometers
            3. Click "Apply Calibration"
            """)
            
            # Get image dimensions
            h, w = st.session_state.preprocessed_image.shape
            
            # Sliders for calibration lines
            left_pos = st.slider(
                "Left marker position (X)", 
                0.0, 1.0, 
                st.session_state.calibration_line_left,
                key="calib_left"
            )
            right_pos = st.slider(
                "Right marker position (X)",
                0.0, 1.0,
                st.session_state.calibration_line_right,
                key="calib_right"
            )
            
            st.session_state.calibration_line_left = left_pos
            st.session_state.calibration_line_right = right_pos
            
            known_um = st.number_input(
                "Known distance (micrometers)",
                min_value=1.0,
                max_value=10000.0,
                value=float(st.session_state.calibration_known_um),
                step=10.0
            )
            st.session_state.calibration_known_um = known_um
            
            # Calculate pixels between markers
            left_x_px = int(left_pos * w)
            right_x_px = int(right_pos * w)
            pixel_distance = abs(right_x_px - left_x_px)
            
            st.info(f"📐 Distance in pixels: **{pixel_distance} px**")
            
            if st.button("✅ Apply Calibration", type="primary"):
                if pixel_distance > 0:
                    um_per_pixel = known_um / pixel_distance
                    st.session_state.calibration_um_per_pixel = um_per_pixel
                    st.success(f"Calibration applied: 1 pixel = {um_per_pixel:.4f} µm")
                else:
                    st.error("Please adjust markers to have non-zero distance")
        
        with col2:
            st.subheader("Material Type")
            
            material_type = st.selectbox(
                "Select material / structure",
                [
                    "Dense ceramic",
                    "Porous ceramic",
                    "Two-phase dense ceramic",
                    "Two-phase porous composite",
                    "Multiphase material",
                    "Metal (polished/etched)"
                ],
                index=["Dense ceramic", "Porous ceramic", "Two-phase dense ceramic", 
                       "Two-phase porous composite", "Multiphase material", 
                       "Metal (polished/etched)"].index(st.session_state.material_type)
            )
            st.session_state.material_type = material_type
            
            st.markdown("---")
            st.caption(f"Selected: **{material_type}**")
            
            if material_type == "Porous ceramic":
                st.info("🔍 Porosity detection will be enabled")
            elif "Two-phase" in material_type:
                st.info("🔍 Second phase detection will be enabled")
            elif material_type == "Metal":
                st.info("🔍 Enhanced edge detection will be applied")
        
        # Display calibration figure
        if st.session_state.raw_image is not None:
            st.subheader("Calibration Preview")
            fig = create_calibration_figure(
                st.session_state.raw_image,
                st.session_state.calibration_line_left,
                st.session_state.calibration_line_right
            )
            st.pyplot(fig)
            plt.close(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("◀️ Back to Stage 1", use_container_width=True):
                st.session_state.stage = 1
                st.rerun()
        with col2:
            if st.button("➡️ Next: Analyze", type="primary", use_container_width=True):
                if st.session_state.calibration_um_per_pixel is not None:
                    st.session_state.stage = 3
                    st.rerun()
                else:
                    st.error("Please calibrate scale first")

# ============================================================================
# STAGE 3: ANALYSIS & SEGMENTATION
# ============================================================================

elif st.session_state.stage == 3:
    st.header("🔬 Stage 3: Microstructure Analysis")
    
    if st.session_state.preprocessed_image is None:
        st.error("No image found. Please start over.")
        st.session_state.stage = 1
        st.rerun()
    
    if st.session_state.calibration_um_per_pixel is None:
        st.error("Calibration not set. Please go back to Stage 2.")
        if st.button("◀️ Back to Stage 2"):
            st.session_state.stage = 2
            st.rerun()
    
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Analysis Parameters")
            
            segmentation_method = st.selectbox(
                "Segmentation method",
                ["Ananyev et al. (classical)", "Watershed", "Adaptive thresholding"]
            )
            
            min_grain_size = st.number_input(
                "Minimum grain size (pixels)",
                min_value=10,
                max_value=500,
                value=50,
                step=10
            )
            
            run_analysis = st.button(
                "🚀 Run Analysis",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            st.info(f"""
            **Analysis configuration:**
            - Material: {st.session_state.material_type}
            - Calibration: {st.session_state.calibration_um_per_pixel:.4f} µm/px
            - Method: {segmentation_method}
            - Min grain size: {min_grain_size} px
            """)
        
        if run_analysis:
            with st.spinner("Analyzing microstructure..."):
                
                # Step 1: Binarization
                if segmentation_method == "Ananyev et al. (classical)":
                    binary = segment_grains_ananyev(st.session_state.preprocessed_image)
                elif segmentation_method == "Watershed":
                    # Otsu threshold first
                    thresh = filters.threshold_otsu(st.session_state.preprocessed_image)
                    binary = (st.session_state.preprocessed_image > thresh).astype(np.uint8)
                    # Clean
                    binary = morphology.remove_small_objects(binary.astype(bool), min_size=50)
                    binary = binary.astype(np.uint8)
                else:  # Adaptive thresholding
                    binary = adaptive_binarization(
                        st.session_state.preprocessed_image,
                        method='adaptive_gaussian',
                        block_size=51,
                        c=5
                    )
                
                # Step 2: Segmentation
                if segmentation_method == "Watershed":
                    segmented = segment_grains_watershed(
                        binary, 
                        st.session_state.preprocessed_image,
                        min_distance=10
                    )
                else:
                    # Label connected components
                    segmented = measure.label(binary.astype(bool))
                
                # Step 3: Extract properties
                props_df = extract_grain_properties(
                    segmented, 
                    st.session_state.calibration_um_per_pixel
                )
                
                # Step 4: Calculate porosity (if applicable)
                porosity = calculate_porosity(binary)
                
                # Step 5: Statistics
                stats = {
                    'total_grains': len(props_df),
                    'mean_grain_area_um2': props_df['area_um2'].mean(),
                    'std_grain_area_um2': props_df['area_um2'].std(),
                    'mean_grain_diameter_um': props_df['eq_diameter_um'].mean(),
                    'std_grain_diameter_um': props_df['eq_diameter_um'].std(),
                    'porosity_percent': porosity,
                    'calibration_um_per_pixel': st.session_state.calibration_um_per_pixel
                }
                
                # Calculate D10, D50, D90
                diameters = props_df['eq_diameter_um'].sort_values()
                stats['D10'] = diameters.quantile(0.10)
                stats['D50'] = diameters.quantile(0.50)
                stats['D90'] = diameters.quantile(0.90)
                stats['uniformity_coefficient'] = stats['D60'] if 'D60' in locals() else stats['D90'] / stats['D10']
                
                st.session_state.segmentation_results = {
                    'binary': binary,
                    'segmented': segmented,
                    'properties': props_df,
                    'stats': stats
                }
                
                st.success("Analysis complete!")
        
        # Display results if available
        if st.session_state.segmentation_results is not None:
            results = st.session_state.segmentation_results
            
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Statistics display
            stats = results['stats']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Grains", stats['total_grains'])
            with col2:
                st.metric("Mean Grain Diameter", f"{stats['mean_grain_diameter_um']:.2f} µm")
            with col3:
                st.metric("Porosity", f"{stats['porosity_percent']:.2f} %")
            with col4:
                st.metric("D50", f"{stats['D50']:.2f} µm")
            
            # Grain size histogram
            st.subheader("📈 Grain Size Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(results['properties']['eq_diameter_um'], bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(stats['D10'], color='red', linestyle='--', label=f'D10 = {stats["D10"]:.2f} µm')
            ax.axvline(stats['D50'], color='green', linestyle='--', label=f'D50 = {stats["D50"]:.2f} µm')
            ax.axvline(stats['D90'], color='blue', linestyle='--', label=f'D90 = {stats["D90"]:.2f} µm')
            ax.set_xlabel('Grain Diameter (µm)')
            ax.set_ylabel('Frequency')
            ax.set_title('Grain Size Distribution')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            
            # Display images
            st.subheader("🖼️ Segmentation Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(results['binary'], caption="Binarized Image", use_container_width=True, clamp=True)
            with col2:
                # Color-coded segmentation
                colored = measure.label(results['segmented'], background=0)
                colored_viz = np.zeros((*colored.shape, 3), dtype=np.uint8)
                for label_id in np.unique(colored):
                    if label_id == 0:
                        continue
                    mask = (colored == label_id)
                    color = np.random.randint(0, 255, 3)
                    colored_viz[mask] = color
                st.image(colored_viz, caption="Segmented Grains", use_container_width=True)
            
            # Data table
            with st.expander("📋 Grain Data Table"):
                st.dataframe(
                    results['properties'].head(20),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                csv = results['properties'].to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "grain_data.csv",
                    "text/csv"
                )
            with col2:
                # Create report
                report = f"""
                MICROSTRUCTURE ANALYSIS REPORT
                ================================
                Material Type: {st.session_state.material_type}
                Calibration: {stats['calibration_um_per_pixel']:.4f} µm/pixel
                
                GRAIN STATISTICS:
                - Total grains: {stats['total_grains']}
                - Mean area: {stats['mean_grain_area_um2']:.2f} µm²
                - Std area: {stats['std_grain_area_um2']:.2f} µm²
                - Mean diameter: {stats['mean_grain_diameter_um']:.2f} µm
                - Std diameter: {stats['std_grain_diameter_um']:.2f} µm
                - D10: {stats['D10']:.2f} µm
                - D50: {stats['D50']:.2f} µm
                - D90: {stats['D90']:.2f} µm
                - Porosity: {stats['porosity_percent']:.2f} %
                """
                st.download_button(
                    "📄 Download Report",
                    report,
                    "analysis_report.txt"
                )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("◀️ Back to Calibration", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()
        with col2:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.segmentation_results = None
                st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<center><small>Microstructure Analyzer v1.0 | Built with Streamlit, OpenCV, scikit-image</small></center>",
    unsafe_allow_html=True
)
