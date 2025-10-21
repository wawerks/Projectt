// background.js

// Create right-click context menu
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "detectImage",
    title: "Detect Image (Crop Face)",
    contexts: ["image"]
  });
  console.log("✅ Context menu created: Detect Image (Crop Face)");
});

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

// Listen for cropped image data
chrome.runtime.onMessage.addListener(async (message) => {
  if (message.action === "croppedImageReady" && (message.imageBlob || message.buffer || message.dataUrl)) {
    console.log("📤 Received cropped image data, preparing to send to backend...");

    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon1.jpg",
      title: "AI Image Detector",
      message: "Analyzing image..."
    });

    try {
      let blob = message.imageBlob;
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

      // ✅ STEP 1: Run Face Detection
      const faceForm = new FormData();
      const faceFile = new File([blob], "face_check.jpg", { type: blob.type || "image/jpeg" });
      faceForm.append("file", faceFile);

      const faceResponse = await fetch("http://127.0.0.1:8000/detect_face", {
        method: "POST",
        body: faceForm
      });

      const faceData = await faceResponse.json();
      if (!faceData.face_detected) {
        chrome.notifications.create({
          type: "basic",
          iconUrl: "icon3.jpg",
          title: "AI Image Detector",
          message: "❌ No human face detected. Please select a valid image."
        });
        return;
      }

      // ✅ STEP 2: Proceed to Deepfake Detection
      

      const formData = new FormData();
      const file = new File([blob], "cropped_face.jpg", { type: blob.type || "image/jpeg" });
      formData.append("file", file);

      const response = await fetch("http://127.0.0.1:8000/classify_image", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      // ✅ Only show the model with the highest confidence
      if (data.individual_results && data.individual_results.length > 0) {
        const highest = data.individual_results.reduce((max, r) =>
          r.confidence > max.confidence ? r : max
        );

        const emoji = highest.label.includes("Real") ? "✅" : "❌";
        const messageText = `${emoji} ${highest.model}: ${highest.label} (${highest.confidence}%)`;

        chrome.notifications.create({
          type: "basic",
          iconUrl: "icon2.jpg",
          title: "Detection Result",
          message: messageText
        });
      } else if (data.error) {
        chrome.notifications.create({
          type: "basic",
          iconUrl: "icon3.jpg",
          title: "AI Image Detector",
          message: "Error: " + data.error
        });
      } else {
        chrome.notifications.create({
          type: "basic",
          iconUrl: "icon3.jpg",
          title: "AI Image Detector",
          message: "Unexpected server response."
        });
      }
    } catch (error) {
      chrome.notifications.create({
        type: "basic",
        iconUrl: "icon3.jpg",
        title: "AI Image Detector",
        message: "Error: " + error.message
      });
    }
  }
});
