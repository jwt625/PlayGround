// Function to play sound
function playSound(url) {
    var audio = new Audio(url);
    audio.play();
  }
  
  // Listen for specific DOM changes or events
  document.addEventListener('click', function(event) {
    // Check if the event target matches specific Onshape operations
    if (event.target.matches('.your-operation-selector')) {
      playSound('your-sound-file.mp3');
    }
  }, false);
  
  // More listeners or DOM mutation observers can be added as needed
  