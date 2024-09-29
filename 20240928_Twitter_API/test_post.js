import express from "express";
import { Client, auth } from "twitter-api-sdk";
import dotenv from "dotenv";
dotenv.config();

const app = express();

const URL = process.env.URL || 'http://127.0.0.1';
const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3000;

/**
 * OAuth 2.0 Settings instructions: 
 * - Use the Twitter Developer Portal to create a Project and an App.
 * - Configure your App by selecting the cog icon next to the App you wish to use.
 * - Click Edit under User authentication settings.
 * - Under OAuth 2.0 Settings, enable OAuth 2.0. Select "Web App" as your Type of App.
 * - Under General Authentication Settings make sure your Callback URI / Redirect URL is the hosted URL of Web App. Save your changes.
 */

const authClient = new auth.OAuth2User({
  client_id: process.env.CLIENT_ID,
  client_secret: process.env.CLIENT_SECRET,
  callback: `${URL}:${PORT}/callback`,
  scopes: ["tweet.read","tweet.write","users.read"],
});
const client = new Client(authClient);
  
const STATE = "my-state";

app.get("/callback", async function (req, res) {
  try {
    const { code, state } = req.query;
    if (state !== STATE) return res.status(500).send("State isn't matching");
    await authClient.requestAccessToken(code);
    res.redirect("/tweets");
  } catch (error) {
    console.log(error);
  }
});
  
app.get("/login", async function (req, res) {
  const authUrl = authClient.generateAuthURL({
    state: STATE,
    code_challenge_method: "s256",
  });
  res.redirect(authUrl);
});

app.get("/revoke", async function (req, res) {
  try {
    const response = await authClient.revokeAccessToken();
    res.send(response);
  } catch (error) {
    console.log(error);
  }
});

app.get("/tweets", async function (req, res) {
  try {
    const response = await client.tweets.createTweet({
    "text": "Test API"
  });
  
  console.log("response", JSON.stringify(response, null, 2));
    res.send(response);
  } catch (error) {
    console.log("tweets error", error);
  }
});

app.listen(PORT, () => {
  console.log(`Go here to login: ${URL}:${PORT}/login`);
});
  