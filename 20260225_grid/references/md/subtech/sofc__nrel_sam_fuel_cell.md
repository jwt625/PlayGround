---
category: sofc
key: nrel_sam_fuel_cell
url: https://samrepo.nrelcloud.org/help/fuelcell.html
final_url: https://samrepo.nrelcloud.org/help/fuelcell.html
retrieved_at_utc: 2026-03-02T04:20:33.961063+00:00
source_type: html
raw_path: references/raw/subtech/sofc/nrel_sam_fuel_cell.html
---

﻿

Fuel Cell

/*! Help+Manual WebHelp 3 Script functions

Copyright (c) 2015-2023 by Tim Green. All rights reserved. Contact: https://www.helpandmanual.com

*/

var pageName = "mainPage";

var hmDevice = {

mobileSleepReload: true,

// Unique project ID

compilehash: (function () {

var outVal = 0, t="SAM HelpUser-defined Project ID";

for (var x = 0; x 0) || (navigator.msMaxTouchPoints > 0));

// Treat all Windows 8 devices as desktops unless in metro mode with touch

hmDevice.tb = (/tablet/.test(hmDevice.agent) && (!/trident/.test(hmDevice.agent) || (hmDevice.w8metro && hmDevice.touch)));

hmDevice.goodandroid = (/android.+?applewebkit\/(?:(?:537\.(?:3[6-9]|[4-9][0-9]))|(?:53[8-9]\.[0-9][0-9])|(?:54[0-9]\.[0-9][0-9]))|android.+?gecko\/[345][0-9]\.\d{1,2} firefox/.test(hmDevice.agent));

hmDevice.deadandroid = (/android.+?applewebkit\/(?:53[0-6]\.\d{1,2})|firefox\/[0-2]\d\.\d{1,2}/.test(hmDevice.agent));

hmDevice.android = (/android/.test(hmDevice.agent) && !hmDevice.deadandroid);

hmDevice.mb = /mobi|mobile/.test(hmDevice.agent);

/* Main Checks */

hmDevice.phone = (hmDevice.mb && !hmDevice.ipad && !hmDevice.tb);

hmDevice.tablet = (hmDevice.ipad || hmDevice.tb || (!hmDevice.phone && hmDevice.android));

hmDevice.aspectRatio = (screen.height / screen.width > 1 ? screen.height / screen.width : screen.width / screen.height).toFixed(2);

hmDevice.narrowTablet = (hmDevice.tablet && hmDevice.aspectRatio > 1.4);

hmDevice.desktop = ((!hmDevice.tablet && !hmDevice.phone));

hmDevice.device = hmDevice.phone ? "phone" : hmDevice.tablet ? "tablet" : hmDevice.desktop ? "desktop" : "default";

hmDevice.fontSize = hmDevice.phone ? hmDevice.baseFontSize * 1.0 : hmDevice.tablet ? hmDevice.baseFontSize * 1.0 : hmDevice.desktop ? hmDevice.baseFontSize : hmDevice.baseFontSize;

hmDevice.maxFontSize = hmDevice.fontSize + hmDevice.fontChangeMax;

hmDevice.minFontSize = hmDevice.fontSize - hmDevice.fontChangeMax;

if (hmDevice.device !== sessionVariable.getPV("hmDevice")) {

sessionVariable.setPV("hmDevice", hmDevice.device);

sessionVariable.setPV("hmFontSize",hmDevice.fontSize.toString());

} else {

var tempSize = sessionVariable.getPV("hmFontSize");

if (tempSize !== null)

hmDevice.fontSize = parseInt(tempSize,10);

else

sessionVariable.setPV("hmFontSize",hmDevice.fontSize.toString());

}

// Expand to full width of container frame if help is embedded in another page

if (hmDevice.embedded && hmDevice.desktop) {

let embedStyle = document.createElement("style");

embedStyle.type = "text/css";

embedStyle.appendChild(document.createTextNode("div#helpwrapper {max-width: 100% !important}"));

document.getElementsByTagName("head")[0].appendChild(embedStyle);

}

hmDevice.pageStylesheet = "./css/hmprojectstyles.css";

var hmSearchActive = true,

hmIndexActive = false;

div#pagewrapper {display: none;}

div#fouc {display: none;}

div#noscript {

display: block;

position: absolute;

top: 0; right: 0; bottom: 0; left: 0;

border: 1rem solid grey;

background-color: #eaeaea;

}

h2#noscriptwarning {

text-align: center;

position: relative;

top: 35%;

color: #333333;

}

Please enable JavaScript to view this site.

-->

SAM Website

LK Script

LK Script

Basic Instructions

LK Reference Manual (PDF)

Sample Scripts

Contact

Contact

Forum

Email

About

About

Release Notes

Website

Zoom Window Out

Larger Text  |  Smaller Text

Hide Page Header

Show Expanding Text

Send Mail Feedback

Save Permalink URL

Navigation: » No topics above this level «

Fuel Cell

Links

Topic Contents

Menu

SAM's Fuel Cell model is based on the Fuel Cell Power Model initially released in 2012 and designed to model molten carbonate, phosphoric acid, and solid oxide fuel cells for "tri-generation" systems that provide heat and electricity to a commercial building or facility and hydrogen for vehicles or for storage to generate electricity. The Fuel Cell Power model was originally implemented in Microsoft Excel workbooks with one workbook for each of the three types of fuel cell. These workbooks and their documentation are available for download from the Fuel Cell Power Model website linked above.

The model is implemented in SAM as the Fuel Cell performance model, available with either the Commercial or PPA Single Owner financial model . The fuel cell converts natural gas, hydrogen, or another fuel into electricity and heat. The commercial model is for a commercial building or facility that uses electricity and heat from the system to reduce energy costs for the commercial operation. The PPA single owner model is for a revenue-generating power generation project.

The fuel cell performance model uses the following input pages to describe the components of a grid-connected tri-generation system:

• Fuel cell model that converts a fuel into electricity, heat, and hydrogen.

• PVWatts for a photovoltaic (PV) system.

• Optional battery storage model for an electric storage system.

• A dispatch model to determine how to operate the combined system.

When you combine the fuel cell model with the Commercial financial model, you must also provide information about the electricity and thermal loads and retail rates for electricity and heat: retail electricity and heat costs:

• An electric load hourly or subhourly profile.

• A thermal load (heat) hourly or subhourly profile.

• Electricity rates are retail rates for fixed, energy, and demand charges with optional time-of-use and tiered rates.

• Thermal rates are flat retail rates for heat purchases and sales.

For the PPA single owner model with battery storage, there is no electric or thermal load because the system is not for a commercial building or facility, but you can specify retail rates to account for the cost of electricity to charge the battery.

System Advisor Model (SAM) Help © National Renewable Energy Laboratory

var fouc=document.getElementById("fouc");

/*! Help+Manual WebHelp 3 Script functions

Copyright (c) 2015-2023 by Tim Green. All rights reserved. Contact: https://www.helpandmanual.com

*/

// Reset font size with preset cookie value if available

(function(){

var tempFontSize = sessionVariable.getPV("hmFontSize"),

baseElement = document.getElementsByTagName("html")[0];

if (tempFontSize !== null) {

baseElement.style.fontSize=tempFontSize + "%";

} else {

baseElement.style.fontSize = hmDevice.baseFontSize + "%";

}

})();

/*** TOGGLE INIT PLACEHOLDER ***/

var moveCheck = null, hmTocLoaded = false, hmTocWidth = 0, xMessage = new xMsg("LANDING TOPIC");

// Post-loading scripts function

function initScripts() {

hmPopupsObject = {};

hmFlags.isHmTopic = true;

hmFlags.idxLoaded = false;

hmFlags.schLoaded = false;

hmFlags.searchSession = false;

hmFlags.tocClicked = false;

hmFlags.layoutTable = false;

hmFlags.searchHighlight = "";

hmFlags.tocInitWidth = 0;

hmFlags.jumpHighlight = false;

hmFlags.hmMainPage = "index.html";

hmFlags.hmTrackHost = "https://sam.nrel.gov/help/";

hmFlags.hmProject = "SAM Help";

hmFlags.hmDefaultPage = "index.html";

hmFlags.hmCurrentPage = "fuelcell.html";

hmFlags.hmTOC = "hmcontent.html";

hmFlags.hmSearchPage = "hmftsearch.html";

hmFlags.hmIndexPage = "hmkwindex.html";

hmFlags.thisTopic = "fuelcell.html";

hmFlags.lastTopic = "index.html";

hmFlags.isEWriter = false;

hmFlags.topicExt = ".html";

hmFlags.defaultExt = ".html";

hmFlags.projName = "SAMhelp";

hmFlags.tDescription = document.querySelector("meta[name='description']").getAttribute("content");

hmFlags.tKeywords = "";

hmFlags.hdFormat = false;

hmFlags.contextID = (function(){

var cntxMatch;

if (/^https??:\/\//im.test(document.location)) {

cntxMatch = /contextid=\d*/.exec(window.location.search.substring(1));

} else {

cntxMatch = /contextid=\d*/.exec(window.location.hash.substring(1));

}

return cntxMatch != null ? cntxMatch[0] : false;

})();

function loadScripts(x) {

$.getScript( scripts[x], function( data, textStatus, jqxhr ) {

if (currentScript 2) {

xMessage.sendObject("hmsearch",{action: "callfunction",fn: "hmDoFtSearch", fa: searchQ});

return;

};

if (event.which == 27) {

searchQ = "";

$(this).val("");

return;

}

$(this).attr("placeholder", "");

xMessage.sendObject("hmsearch",{action: "callfunction",fn: "loadSquery", fa: searchQ});

});

hmWebHelp.hmMainPageInit();

}

}); // $getScript

} // loadScripts()

// Force HTML4 mode in History framework for local WebHelp (required for Chrome, use globally for consistency)

if (document.location.protocol.substr(0,4) !== "http") {

window.History = {options: {html4Mode: true} };

}

var currentScript = 0;

var scripts = ["./js/helpman_settings.js", "./js/jquery.scrollTomin.js", "./js/jquery.history.js", "./js/hm_webhelp.js", "./js/HM_CLASSICMENU.js" ];

if (hmFlags.contextID) scripts.unshift("./js/hmcontextids.js");

var jScript = document.getElementById("jqscript");

jScript.src = "./js/jquery.js";

var loadCheck = setInterval(function() {

if (window.jQuery) {

clearInterval(loadCheck);

$.ajaxPrefilter( "json script", function(options) {options.crossDomain = true;});

if (hmDevice.embedded && hmDevice.desktop) $("li#toggle_fullscreen").show();

loadScripts(0);

/* Title Bar Text */

var titleBarText = "Fuel Cell";

switch ("topic") {

case "project":

titleBarText = "SAM Help";

break;

case "project_topic":

titleBarText = "SAM Help" + " \> " + "Fuel Cell";

break;

}

titleBarText = $(" ").html(titleBarText).text();

$("title").text(titleBarText);

}

},100);

}

if (window.addEventListener)

window.addEventListener("load", initScripts, false);

else if (window.attachEvent)

window.attachEvent("onload", initScripts);

else window.onload = initScripts;

helpman_mailrecipient = 'sam.support@nrel.gov';

hmGetUrlParams = function() {/*!

* Enter code to save URL parameters to session variables inside the if clause below.

* This will be executed when your WebHelp is first opened and the session variables

* you save here will then be available. See the session variables documentation in

* the V3 Responsive Skins chapter of the help for detailed instructions.

*/

if (hmWebHelp.userParams && hmWebHelp.userParams.paramsCount > 0) {

}

}

/*!

* Enter the JS function/s you want to be able to call after the main page

* or individual topics load in this file. The functions are made available

* by this file but called by the POSTLOAD_TOPICFUNC and POSTLOAD_PAGEFUNC

* variables.

*

* Nothing in this file should be executed on loading! All execution is

* initiated with a single statement stored in the POSTLOAD_PAGEFUNC and

* POSTLOAD_TOPICFUNC variables, in the General Settings variables group.

*

* POSTLOAD_PAGEFUNC: Statement to be executed ONCE on loading the main page.

* POSTLOAD_TOPICFUNC: Statement to be executed on EVERY page load

*

* The statement/s you enter POSTLOAD_PAGEFUNC will be executed after

* all page components and assets have loaded, including images.

*

* The statement/s in POSTLOAD_TOPICFUNC are executed after topic content

* has been dynamically replaced in the page when the user browses to a new

* topic.

*/

Keyboard Navigation

F7 for caret browsing

Hold ALT and press letter

This Info:

ALT+q

Page Header:

ALT+h

Topic Header:

ALT+t

Topic Body:

ALT+b

Contents:

ALT+c

Search:

ALT+s

Exit Menu/Up:

ESC
