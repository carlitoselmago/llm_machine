"use strict";

const { app, BrowserWindow } = require("electron");
const { start, PORT } = require("./server.js");

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 900,
    title: "Gerineldo",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  mainWindow.loadURL(`http://localhost:${PORT}`);
  mainWindow.setMenuBarVisibility(false);
  mainWindow.on("closed", () => { mainWindow = null; });
}

app.whenReady().then(async () => {
  await start();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  app.quit();
});
