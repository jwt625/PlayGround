import { travelTimeClient, readJSONFile } from './travel-time-module';

// Call the readJSONFile function to fetch and set APP_ID and APP_KEY
readJSONFile().then(() => {
  // Now, you can use travelTimeClient with the retrieved APP_ID and APP_KEY
  console.log(travelTimeClient);
});

