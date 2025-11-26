modal_css = """
/* Hide the default auto-generated close button */
.modal-container .close {
    display: none !important;
}

/* Now style your custom close button normally */
#close-btn {
    position: absolute !important;
    top: 15px;
    right: 15px;
    z-index: 100;
    background: rgba(0,0,0,0.6);
    color: white;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.2s ease;
}

#close-btn:hover {
    background: white;
    color: black;
    border-color: white;
}

/* Modal card */
.modal-body {
    background-color: #181818 !important;
    border-radius: 12px !important;
    max-width: 500px !important;
    width: 90vw !important;
    max-height: 90vh !important;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    position: relative;
    box-shadow: 0 25px 50px rgba(0,0,0,0.9);
}

/* Modal image */
.modal-body img {
    width: 100%;
    height: 300px;
    object-fit: cover;
}

/* Title & Description */
.modal-body h2 {
    padding: 20px;
    margin: 0;
    color: white;
    font-size: 1.5rem;
}
.modal-body p {
    padding: 0 20px 20px 20px;
    color: #ccc;
    font-size: 0.95rem;
    line-height: 1.5;
    overflow-y: auto;
    max-height: 200px;
}
"""


custom_css = """
/* --- GLOBAL THEME --- */
body, .gradio-container { 
    background-color: #141414 !important; /* Netflix Black */
    color: #e5e5e5 !important; 
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
}

/* --- HEADER --- */
.header { 
    text-align: center; 
    padding: 60px 20px 20px 20px; 
}
.header h1 { 
    font-size: 3.5rem; 
    font-weight: 700; 
    color: #e50914; /* Netflix Red */
    margin: 0;
}
.header p { 
    color: #a3a3a3; 
    font-size: 1.2rem; 
    margin-top: 10px; 
}

/* --- WIDE & INVISIBLE INPUT CONTAINER --- */
.inputs { 
    max-width: 1200px !important; /* Make it Wide */
    width: 100% !important;
    margin: 0 auto 50px auto !important; 
    background: transparent !important; /* Into the background */
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    gap: 15px;
}

/* Style the actual input boxes to pop against the background */
.gr-textbox, .gr-dropdown {
    background-color: rgba(50, 50, 50, 0.8) !important;
    border: 1px solid #444 !important;
    border-radius: 4px !important;
}
/* Text inside inputs */
.gr-textbox input, .gr-dropdown span {
    color: white !important;
    font-size: 1.1rem;
}

.custom-btn {
    background: #e50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: bold;
    font-size: 1rem;
    height: 100%; 
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}
.custom-btn:hover {
    background: #f40612 !important;
    box-shadow: 0 0 15px rgba(229, 9, 20, 0.6);
}

/* --- GALLERY --- */
.gr-gallery {
    padding: 20px;
}
.gallery-item img {
    border-radius: 4px;
    transition: transform 0.3s ease;
}
.gallery-item:hover img {
    transform: scale(1.05);
    z-index: 10;
}

""" + modal_css