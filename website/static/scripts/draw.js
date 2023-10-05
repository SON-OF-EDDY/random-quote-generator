
// JavaScript code for drawing on the canvas
const canvas = document.getElementById("drawingCanvas");
const context = canvas.getContext("2d");
context.strokeStyle = 'white';
context.lineWidth = 25;
let isDrawing = false;

// Event listeners for both mouse and touch events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("touchstart", startDrawing);

canvas.addEventListener("mousemove", draw);
canvas.addEventListener("touchmove", draw);

canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("touchend", stopDrawing);

canvas.addEventListener("mouseout", stopDrawing);

// Touch events need to prevent the default touch actions to work properly
canvas.addEventListener("touchstart", (event) => {
    event.preventDefault();
});
canvas.addEventListener("touchmove", (event) => {
    event.preventDefault();
});

function startDrawing(event) {
    isDrawing = true;
    context.beginPath();

    if (event.type === "touchstart") {
        const touch = event.touches[0];
        const x = touch.clientX - canvas.getBoundingClientRect().left;
        const y = touch.clientY - canvas.getBoundingClientRect().top;
        context.moveTo(x, y);
    } else {
        const x = event.clientX - canvas.getBoundingClientRect().left;
        const y = event.clientY - canvas.getBoundingClientRect().top;
        context.moveTo(x, y);
    }
}

function draw(event) {
    if (!isDrawing) return;

    if (event.type === "touchmove") {
        event.preventDefault();
        const touch = event.touches[0];
        const x = touch.clientX - canvas.getBoundingClientRect().left;
        const y = touch.clientY - canvas.getBoundingClientRect().top;
        context.lineTo(x, y);
    } else {
        const x = event.clientX - canvas.getBoundingClientRect().left;
        const y = event.clientY - canvas.getBoundingClientRect().top;
        context.lineTo(x, y);
    }

    context.stroke();
}

function stopDrawing() {
    isDrawing = false;
    context.closePath();
}


// Function to save the canvas content as an image with a white background
const saveButton = document.getElementById("saveButton");
//const savedImage = document.getElementById("savedImage");
const tempCanvas = document.getElementById("tempCanvas");
//const imageDataURLInput = document.getElementById("imageDataURL"); // Added

saveButton.addEventListener("click", () => {
    // Create a temporary context for the temporary canvas
    const tempContext = tempCanvas.getContext("2d");
    const predictionTag = document.getElementById('prediction-tag');

    // Draw a white background on the temporary canvas
    tempContext.fillStyle = "#000000"; // White color
    tempContext.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Draw the existing canvas content on top of the white background
    tempContext.drawImage(canvas, 0, 0);

    // Capture the temporary canvas content as an image URL
    const finalImageDataURL = tempCanvas.toDataURL("image/png");
    //savedImage.src = finalImageDataURL;

    //imageDataURLInput.value = finalImageDataURL;

    var imageUrl = finalImageDataURL;

    fetch(`/process_image/?image_url=${encodeURIComponent(imageUrl)}`)
                .then(response => response.json())
                .then(data => {
                    // Handle the response from the Django view
                    console.log(data);
                    predictionTag.innerHTML = `Prediction: ${data['prediction']}`;
                    // You can display a success message or perform other actions here
                })
                .catch(error => {
                    console.log('some error...');
                });
});


// Function to clear the canvas
const clearButton = document.getElementById("clearButton");
clearButton.addEventListener("click", () => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    //savedImage.src = '';
    //savedImage.classList.remove("image-with-border");
    //downloadLink.hidden = true;
});