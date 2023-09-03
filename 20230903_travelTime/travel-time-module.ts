// import { TravelTimeClient } from 'traveltime-api';
import {
  TimeMapRequestArrivalSearch,
  TimeMapRequestDepartureSearch,
  TravelTimeClient
} from 'traveltime-api';

var APP_ID : string
var APP_KEY : string

function readJSONFile(): Promise<void>  {
    return fetch('apikey.json') // Replace 'data.json' with your JSON file's URL or path
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

readJSONFile()

const travelTimeClient = new TravelTimeClient({
  applicationId: APP_ID,
  apiKey: APP_KEY,
});


export { travelTimeClient, readJSONFile };

// const departure_search: TimeMapRequestDepartureSearch = {
//   id: 'public transport from Trafalgar Square',
//   departure_time: new Date().toISOString(),
//   travel_time: 900,
//   coords: { lat: 51.507609, lng: -0.128315 },
//   transportation: { type: 'public_transport' },
//   properties: ['is_only_walking'],
// };

// const arrival_search: TimeMapRequestArrivalSearch = {
//   id: 'public transport to Trafalgar Square',
//   arrival_time: new Date().toISOString(),
//   travel_time: 900,
//   coords: { lat: 51.507609, lng: -0.128315 },
//   transportation: { type: 'public_transport' },
//   range: { enabled: true, width: 3600 },
// };

// travelTimeClient.timeMap({
//   departure_searches: [departure_search],
//   arrival_searches: [arrival_search],
// }).then((data) => console.log(data))
//   .catch((e) => console.error(e));


