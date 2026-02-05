import plotly.graph_objects as go
import numpy as np
from skimage import measure


def plot_3d_interactive(volume, mask, threshold=0.1):
    """
    Renders an interactive 3D visualization of brain MRI with tumor segmentation.

    :param volume: 4D array (H, W, D, channels) - MRI volume
    :param mask: segmentation mask
    :param threshold: isovalue for brain surface extraction, defaults to 0.1
    :type threshold: float, optional

    Use the legend to toggle layers on/off.
    """
    meshes = []

    # Extract brain surface from T1 channel
    brain_volume = volume[:, :, :, 1]
    brain_mesh = _create_brain_mesh(brain_volume, threshold)
    if brain_mesh:
        meshes.append(brain_mesh)

    # Convert one-hot encoded mask to class indices
    if mask.ndim == 4:
        mask = np.argmax(mask, axis=-1)

    # Add tumor class meshes
    tumor_meshes = _create_tumor_meshes(mask)
    meshes.extend(tumor_meshes)

    # Create and configure figure
    fig = go.Figure(data=meshes)
    _configure_layout(fig)
    fig.show()


def _create_brain_mesh(brain_volume, threshold):
    """
    Creates a semi-transparent 3D mesh of the brain surface.

    Uses marching cubes algorithm to extract an isosurface from the T1 MRI channel.
    Returns None if surface extraction fails due to insufficient contrast.

    :param brain_volume: single channel MRI volume
    :param threshold: isovalue for surface extraction

    :return: brain surface mesh object or None
    """

    try:
        verts, faces, _, _ = measure.marching_cubes(
            brain_volume,
            level=threshold,
            step_size=2
        )

        return go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.1,
            color='lightgray',
            name='Brain Surface',
            showlegend=True,
            hoverinfo='name'
        )
    except (ValueError, RuntimeError):
        print("Warning: Could not generate brain surface (insufficient contrast)")
        return None


def _create_tumor_meshes(mask):
    """
    Creates colored 3D meshes for each tumor class in the segmentation mask.

    Generates separate meshes for necrotic core (class 1), peritumoral edema (class 2),
    and enhancing tumor (class 3) using distinct colors.

    :param mask: segmentation mask with class indices (0-3)
    :return: tumor region meshes
    """

    CLASS_CONFIG = {
        1: {'color': '#FF4444', 'name': 'Necrotic Core', 'opacity': 0.9},
        2: {'color': '#44FF44', 'name': 'Peritumoral Edema', 'opacity': 0.4},
        3: {'color': '#FFFF44', 'name': 'Enhancing Tumor', 'opacity': 0.7}
    }

    meshes = []
    for class_id in [2, 3, 1]:
        if class_id not in np.unique(mask):
            continue

        config = CLASS_CONFIG[class_id]
        binary_mask = (mask == class_id).astype(float)

        try:
            verts, faces, _, _ = measure.marching_cubes(
                binary_mask,
                level=0.5,
                step_size=1
            )

            mesh = go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=config['opacity'],
                color=config['color'],
                name=config['name'],
                showlegend=True,
                hoverinfo='name'
            )
            meshes.append(mesh)
        except (ValueError, RuntimeError):
            continue

    return meshes



def _configure_layout(fig):
    """
    Applies visual styling and layout configuration to the 3D figure.

    Configures title, dimensions, camera position, legend, axis visibility,
    and color scheme for the interactive visualization.

    :param fig: figure object to configure
    :return: None - modifies figure in place
    """
    fig.update_layout(
        title={
            'text': "3D Brain Tumor Visualization",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        width=1000,
        height=700,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#34495e",
            borderwidth=1,
            font=dict(size=12)
        ),
        paper_bgcolor='#ecf0f1',
        plot_bgcolor='#ecf0f1'
    )
