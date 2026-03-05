// content.js — runs on every YouTube page
// Its only job: tell the popup what the current video URL is

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getVideoUrl") {
    sendResponse({ url: window.location.href });
  }
});