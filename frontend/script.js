let selectedFolderPath = "";
let folderLength = 0
let seg = false

async function chooseFolder() {
    const folder = await window.pywebview.api.choose_folder();
    if (folder) {
        selectedFolderPath = folder;
        selectedFolderName = folder.split(/[\\/]/).pop();
        document.getElementById('folderPathDisplay').innerText = folder;
        await listFiles(folder);
    }
}

async function listFiles(folder) {
    const files = await window.pywebview.api.list_files(folder);
    folderLength = files.length
    const list = document.getElementById('fileList');
    list.innerHTML = '';

    if (files.length === 0) {
        const li = document.createElement('li');
        li.innerText = "No image files found.";
        list.appendChild(li);
    } else {
        files.forEach(file => {
            const li = document.createElement('li');
            li.innerText = file;
            li.onclick = () => previewImage(file);
            list.appendChild(li);
        });
    }

    const img = document.getElementById('preview');
    const placeholder = document.getElementById('placeholder-text');
    img.src = "";
    img.style.display = 'none';
    placeholder.style.display = 'block';
}


function previewImage(filename) {
    const img = document.getElementById('preview');
    const placeholder = document.getElementById('placeholder-text');

    if (!selectedFolderName) {
        console.error("❌ selectedFolderName not set");
        return;
    }

    const httpPath = `/${selectedFolderName}/${encodeURIComponent(filename)}`;
    console.log("🖼️ Previewing image:", httpPath);

    img.src = httpPath;
    img.style.display = 'block';
    placeholder.style.display = 'none';
}


async function process() {
    const prompt = document.getElementById('prompt').value.trim();
    const mode = document.getElementById('mode').value;
    const startBtn = document.querySelector('button[onclick="process()"]');
    const cancelBtn = document.getElementById('cancelBtn');

    if (!selectedFolderPath) {
        alert("Please choose a folder first.");
        return;
    }

    if (!prompt) {
        alert("Please enter a prompt.");
        return;
    }

    // const files = await window.pywebview.api.list_files(selectedFolderPath);
    // folderLength = files.length
    initProgressBar(folderLength);

    // === UI: Disable Start button, Show Cancel button ===
    startBtn.disabled = true;
    startBtn.innerText = "Processing...";
    cancelBtn.style.display = "inline-block";
    cancelBtn.disabled = false;
    cancelRequested = false;

    try {

        if (mode == "bbox") {
            seg = false
        } else {
            seg = true
        }
        document.getElementById("processedList").innerHTML = "";
        const processedFolder = await window.pywebview.api.process_image(selectedFolderPath, prompt, mode);
        if (processedFolder) {
            await listProcessedFiles(processedFolder);
            alert("✅ Processing complete!");
        } else {
            alert("❌ Processing failed.");
        }
    } catch (err) {
        console.error(err);
        alert("❌ Error occurred during processing.");
    } finally {

        hideProgressBar();
        resetProgressBar();

    }

    // === UI: Re-enable Start button, Hide Cancel button ===
    startBtn.disabled = false;
    startBtn.innerText = "Start Processing";
    cancelBtn.style.display = "none";
    cancelBtn.disabled = true;
}

let cancelRequested = false;


function cancelProcess() {
    cancelRequested = true;
    window.pywebview.api.cancel_processing();
    const cancelBtn = document.getElementById('cancelBtn');
    cancelBtn.disabled = true;
    alert("❗️ Cancellation requested. Will stop after current image.");
}


function addToProcessedList(filename) {
    const list = document.getElementById('processedList');
    const li = document.createElement('li');
    li.innerText = filename;

    li.onclick = () => {
        let httpPath;
        if (seg) {
            httpPath = `output_seg/${filename}`;
        } else {
            httpPath = `output_bbox/${filename}`;
        }
        console.log("filename is:", httpPath);
        previewImage(httpPath);
    };

    list.appendChild(li);
}


async function listProcessedFiles(folder) {
    const files = await window.pywebview.api.list_files(folder);
    const list = document.getElementById('processedList');
    list.innerHTML = '';

    if (files.length === 0) {
        const li = document.createElement('li');
        li.innerText = "No processed files found.";
        list.appendChild(li);
        return;
    }

    for (const file of files) {
        const index = files.indexOf(file);
        const li = document.createElement('li');
        li.innerText = file;
        li.onclick = () => {
            let httpPath;
            if (seg) {
                httpPath = `output_seg/${file}`;
            } else {
                httpPath = `output_bbox/${file}`;
            }
            console.log("filename is:", httpPath);
            previewImage(httpPath);

        }

        list.appendChild(li);

        if (index === 0) {
            let httpPath;
            if (seg) {
                httpPath = `output_seg/${file}`;
            } else {
                httpPath = `output_bbox/${file}`;
            }
            console.log("filename is:", httpPath);
            previewImage(httpPath);
        }
    }
}

function initProgressBar(total) {
    const bar = document.getElementById("progressBar");
    bar.style.width = "0%";
    bar.dataset.total = total;
    bar.dataset.current = 0;
    document.getElementById("progressBarContainer").style.display = "block";
}

function updateProgressBar() {
    const bar = document.getElementById("progressBar");
    const total = parseInt(bar.dataset.total);
    let current = parseInt(bar.dataset.current);
    current += 1;
    bar.dataset.current = current;
    const percent = Math.round((current / total) * 100);
    bar.style.width = `${percent}%`;
}

function hideProgressBar() {
    document.getElementById("progressBarContainer").style.display = "none";
}

function resetProgressBar() {
    const bar = document.getElementById("progressBar");
    bar.style.width = "0%";
    bar.dataset.total = 0;
    bar.dataset.current = 0;
}
