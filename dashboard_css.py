modal_css = """
/* --- FULL SCREEN OVERLAY (The 'Container' must be full screen) --- */
.modal-container {
    position: fixed !important;
    top: 0;
    left: 0;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(0, 0, 0, 0.85) !important; 
    display: flex !important;
    align-items: center !important; /* Vertically center the modal-block */
    justify-content: center !important; /* Horizontally center the modal-block */
    z-index: 9999;
    backdrop-filter: blur(6px);
    transition: opacity 0.3s ease;
}

/* Show modal */
.modal-container.active {
    opacity: 1;
    pointer-events: auto;
}

/* --- WRAPPER BLOCK: Shrinks to fit the inner content (modal-body) --- */
.modal-block {
    max-width: fit-content !important; 
    max-height: fit-content !important;
    width: fit-content !important;
    height: fit-content !important;
    
    background: transparent !important; /* Color is now transparent/invisible */
    padding: 0 !important; 
    border: none !important; 
    
    overflow: hidden;
    display: flex; 
    align-items: center;
    justify-content: center;
    margin: 0 !important;
}

/* --- MODAL CARD (The 'Body' is the content you see) --- */
.modal-body {
    position: relative !important;
    background-color: #181818 !important; /* Card Color */
    border-radius: 12px !important;
    
    /* Define the size of the card itself */
    max-width: 450px !important; /* Narrower for taller look */
    width: 90vw !important; 
    max-height: 90vh !important;
    height: fit-content !important; 
    
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 0 40px rgba(0,0,0,1);
    transform: scale(0.8);
    transition: transform 0.3s ease;
}

/* Animate modal open */
.modal-container.active .modal-body {
    transform: scale(1);
}

/* --- CLOSE BUTTON --- */
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

/* --- IMAGE --- */
.modal-body img {
    width: 100%;
    height: 350px; 
    object-fit: cover;
    transition: transform 0.3s ease;
}
.modal-body img:hover {
    transform: scale(1.05);
}

/* --- TITLE & DESCRIPTION --- */
.modal-body h2 {
    padding: 20px;
    margin: 0;
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
}
.modal-body p {
    padding: 0 20px 20px 20px;
    color: #ccc;
    font-size: 0.95rem;
    line-height: 1.5;
    overflow-y: auto;
    max-height: 300px; 
}

/* Hide default Gradio close button */
.modal-container .close {
    display: none !important;
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