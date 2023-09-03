import { TravelTimeClient } from 'traveltime-api';

var APP_ID
var APP_KEY

function readJSONFile() {
    fetch('apikey.json') // Replace 'data.json' with your JSON file's URL or path
      .then(response => response.json())
      .then(data => {
        // Now 'data' contains the parsed JSON object
        APP_ID =  data.APP_ID;
        APP_KEY =  data.APP_KEY;
      })
      .catch(error => {
        console.error('Error reading JSON file:', error);
      });
  }


const travelTimeClient = new TravelTimeClient({
  applicationId: 'YOUR_APP_ID',
  apiKey: 'YOUR_APP_KEY',
});

