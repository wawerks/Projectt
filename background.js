// ======================================================
// background.js — Enhanced (Aligned with Option 1 Server Logic)
// ======================================================

// Create right-click context menu
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "detectImage",
    title: "Detect Image (Crop Face)",
    contexts: ["image"]
  });
  console.log("✅ Context menu created: Detect Image (Crop Face)");
});

// Helper to show notifications
function showNotification(icon, title, message) {
  chrome.notifications.create({
    type: "basic",
    iconUrl: icon,
    title,
    message
  });
}

// Handle context menu click
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "detectImage" && info.srcUrl) {
    console.log("🖼️ Sending START_CROPPING to content.js for:", info.srcUrl);

    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (imageUrl) => {
        window.postMessage({ type: "START_CROPPING", imageUrl }, "*");
      },
      args: [info.srcUrl]
    });
  }
});

// Listen for cropped image data from content.js
chrome.runtime.onMessage.addListener(async (message) => {
  if (message.action !== "croppedImageReady") return;

  console.log("📤 Received cropped image data, preparing to send to backend...");
  showNotification("icon1.jpg", "AI Image Detector", "Analyzing image...");

  try {
    let blob = message.imageBlob;

    // Decode if necessary
    if (!blob && message.buffer) {
      const uint8 = new Uint8Array(message.buffer);
      blob = new Blob([uint8], { type: message.type || "image/jpeg" });
    }
    if (!blob && message.dataUrl) {
      const [meta, base64] = message.dataUrl.split(",");
      const mimeMatch = /data:([^;]+);base64/.exec(meta);
      const mime = (mimeMatch && mimeMatch[1]) || "image/jpeg";
      const binary = atob(base64);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
      blob = new Blob([bytes], { type: mime });
    }

    if (!blob || (blob.size !== undefined && blob.size === 0)) {
      throw new Error("Received empty image blob after decoding.");
    }

    // STEP 1: Face Detection
    const faceForm = new FormData();
    faceForm.append("file", new File([blob], "face_check.jpg", { type: blob.type || "image/jpeg" }));

    const faceResponse = await fetch("http://127.0.0.1:8000/detect_face", {
      method: "POST",
      body: faceForm,
    });

    if (!faceResponse.ok) throw new Error("Face detection request failed.");
    const faceData = await faceResponse.json();

    if (!faceData.face_detected) {
      showNotification("icon3.jpg", "AI Image Detector", "❌ No human face detected. Please select a valid image.");
      return;
    }

    // STEP 2: Deepfake Detection
    const formData = new FormData();
    formData.append("file", new File([blob], "cropped_face.jpg", { type: blob.type || "image/jpeg" }));

    const response = await Promise.race([
      fetch("http://127.0.0.1:8000/classify_image", { method: "POST", body: formData }),
      new Promise((_, reject) => setTimeout(() => reject(new Error("Connection timed out.")), 10000))
    ]);

    if (!response.ok) throw new Error("Deepfake detection request failed.");

    const data = await response.json();

    // Handle new server response format
    if (data.final_decision) {
      const { final_label, real_confidence, fake_confidence } = data.final_decision;

      let emoji = "❌";
      if (final_label.includes("Real")) emoji = "✅";
      else if (final_label.includes("Uncertain")) emoji = "⚠️";

      const msg = `${emoji} ${final_label}\nReal: ${real_confidence}% | Fake: ${fake_confidence}%`;

      showNotification("icon2.jpg", "Detection Result", msg);
    } else if (data.error) {
      showNotification("icon3.jpg", "AI Image Detector", "Error: " + data.error);
    } else {
      showNotification("icon3.jpg", "AI Image Detector", "Unexpected server response.");
    }
  } catch (error) {
    console.error("❌ Error in background.js:", error);
    showNotification("icon3.jpg", "AI Image Detector", "Error: " + error.message);
  }
});
