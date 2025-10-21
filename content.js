// content.js

// Wait for context menu action from background.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "detectImage") {
    showCropper(request.imageUrl);
  }
});

// Also support in-page messages posted via chrome.scripting (background.js)
window.addEventListener("message", (event) => {
  if (event.source === window && event.data && event.data.type === "START_CROPPING") {
    showCropper(event.data.imageUrl);
  }
});

async function showCropper(imageUrl) {
  // Remove existing cropper if open
  const oldOverlay = document.getElementById("ai-detector-overlay");
  if (oldOverlay) oldOverlay.remove();

  // Create overlay
  const overlay = document.createElement("div");
  overlay.id = "ai-detector-overlay";
  Object.assign(overlay.style, {
    position: "fixed",
    top: 0,
    left: 0,
    width: "100vw",
    height: "100vh",
    backgroundColor: "rgba(0,0,0,0.8)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 999999,
    flexDirection: "column"
  });

  // Create image
  const img = document.createElement("img");
  img.src = imageUrl;
  img.id = "image-to-crop";
  img.style.maxWidth = "80vw";
  img.style.maxHeight = "80vh";
  img.style.border = "3px solid white";
  overlay.appendChild(img);

  // Buttons
  const btnContainer = document.createElement("div");
  btnContainer.style.marginTop = "15px";

  const cropBtn = document.createElement("button");
  cropBtn.innerText = "Crop & Detect";
  cropBtn.style.marginRight = "10px";
  cropBtn.style.padding = "8px 15px";
  cropBtn.style.border = "none";
  cropBtn.style.borderRadius = "8px";
  cropBtn.style.cursor = "pointer";
  cropBtn.style.background = "#4CAF50";
  cropBtn.style.color = "#fff";

  const cancelBtn = document.createElement("button");
  cancelBtn.innerText = "Cancel";
  cancelBtn.style.padding = "8px 15px";
  cancelBtn.style.border = "none";
  cancelBtn.style.borderRadius = "8px";
  cancelBtn.style.cursor = "pointer";
  cancelBtn.style.background = "#f44336";
  cancelBtn.style.color = "#fff";

  btnContainer.appendChild(cropBtn);
  btnContainer.appendChild(cancelBtn);
  overlay.appendChild(btnContainer);

  document.body.appendChild(overlay);

  // Initialize cropper
  const cropper = new Cropper(img, {
    aspectRatio: 1,
    viewMode: 1,
    guides: true,
    dragMode: "move",
    zoomable: true,
    background: false
  });

  // Button actions
  cancelBtn.addEventListener("click", () => {
    cropper.destroy();
    overlay.remove();
  });

  cropBtn.addEventListener("click", async () => {
    const croppedCanvas = cropper.getCroppedCanvas({ width: 256, height: 256 });
    overlay.innerHTML = "<p style='color:white;'>⏳ Processing...</p>";

    // Use a Data URL for robust structured cloning across service workers
    const dataUrl = croppedCanvas.toDataURL("image/jpeg", 0.92);
    chrome.runtime.sendMessage({ action: "croppedImageReady", dataUrl });

    // Clean up the UI; background will notify the user about progress and results
    cropper.destroy();
    overlay.remove();
  });
}
