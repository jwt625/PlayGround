---
category: compute
key: google_dram_errors
url: https://research.google/pubs/dram-errors-in-the-wild-a-large-scale-field-study/
final_url: https://research.google/pubs/dram-errors-in-the-wild-a-large-scale-field-study/
retrieved_at_utc: 2026-03-02T05:06:22.122655+00:00
source_type: html
raw_path: references/raw/subtech/compute/google_dram_errors.html
---

DRAM Errors in the Wild: A Large-Scale Field Study

function glueCookieNotificationBarLoaded() {

(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':

new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],

j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=

'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);

})(window,document,'script','dataLayer','GTM-K8QBZ7Q');

}

Jump to Content

Research

Research

Who we are

Back to

Who we are

menu

Defining the technology of today and tomorrow.

Philosophy

We strive to create an environment conducive to many different types of research across many different time scales and levels of risk.

Learn more about our Philosophy

Learn more

Philosophy

People

Our researchers drive advancements in computer science through both fundamental and applied research.

Learn more about our People

Learn more

People

Research areas

Back to

Research areas

menu

Research areas

Explore all research areas

Research areas

Back to

Research areas

menu

Explore all research areas

Foundational ML & Algorithms

Algorithms & Theory

Data Management

Data Mining & Modeling

Information Retrieval & the Web

Machine Intelligence

Machine Perception

Machine Translation

Natural Language Processing

Speech Processing

Foundational ML & Algorithms

Back to

Foundational ML & Algorithms

menu

Algorithms & Theory

Data Management

Data Mining & Modeling

Information Retrieval & the Web

Machine Intelligence

Machine Perception

Machine Translation

Natural Language Processing

Speech Processing

Computing Systems & Quantum AI

Distributed Systems & Parallel

Computing

Hardware & Architecture

Mobile Systems

Networking

Quantum Computing

Robotics

Security, Privacy, & Abuse

Prevention

Software Engineering

Software Systems

Computing Systems & Quantum AI

Back to

Computing Systems & Quantum AI

menu

Distributed Systems & Parallel

Computing

Hardware & Architecture

Mobile Systems

Networking

Quantum Computing

Robotics

Security, Privacy, & Abuse

Prevention

Software Engineering

Software Systems

Science, AI & Society

Climate & Sustainability

Economics & Electronic Commerce

Education Innovation

General Science

Health & Bioscience

Human-Computer Interaction and Visualization

Responsible AI

Science, AI & Society

Back to

Science, AI & Society

menu

Climate & Sustainability

Economics & Electronic Commerce

Education Innovation

General Science

Health & Bioscience

Human-Computer Interaction and Visualization

Responsible AI

Our work

Back to

Our work

menu

Projects

We regularly open-source projects with the broader research community and apply our developments to Google products.

Learn more about our Projects

Learn more

Projects

Publications

Publishing our work allows us to share ideas and work collaboratively to advance the field of computer science.

Learn more about our Publications

Learn more

Publications

Resources

We make products, tools, and datasets available to everyone with the goal of building a more collaborative ecosystem.

Learn more about our Resources

Learn more

Resources

Programs & events

Back to

Programs & events

menu

Shaping the future, together.

Collaborate with us

Student programs

Supporting the next generation of researchers through a wide range of programming.

Learn more about our Student programs

Learn more

Student programs

Faculty programs

Participating in the academic research community through meaningful engagement with university faculty.

Learn more about our Faculty programs

Learn more

Faculty programs

Conferences & events

Connecting with the broader research community through events is essential for creating progress in every aspect of our work.

Learn more about our Conferences & events

Learn more

Conferences & events

Collaborate with us

Careers

Blog

Search

Home

Publications

DRAM Errors in the Wild: A Large-Scale Field Study

Bianca Schroeder

Eduardo Pinheiro

Wolf-Dietrich Weber

SIGMETRICS (2009)

Google Scholar

Copy Bibtex

Abstract

Errors in dynamic random access memory (DRAM) are a common form of hardware failure in modern compute clusters. Failures are costly both in terms of hardware replacement costs and service disruption. While a large body of work exists on DRAM in laboratory conditions, little has been reported on real DRAM failures in large production clusters. In this paper, we analyze measurements of memory errors in a large fleet of commodity servers over a period of 2.5 years. The collected data covers multiple vendors, DRAM capacities and technologies, and comprises many millions of DIMM days. The goal of this paper is to answer questions such as the following: How common are memory errors in practice? What are their statistical properties? How are they affected by external factors, such as temperature and utilization, and by chip-specific factors, such as chip density, memory technology and DIMM age? We find that DRAM error behavior in the field differs in many key aspects from commonly held assumptions. For example, we observe DRAM error rates that are orders of magnitude higher than previously reported, with 25,000 to 70,000 errors per billion device hours per Mbit and more than 8\% of DIMMs affected by errors per year. We provide strong evidence that memory errors are dominated by hard errors, rather than soft errors, which previous work suspects to be the dominant error mode. We find that temperature, known to strongly impact DIMM error rates in lab conditions, has a surprisingly small effect on error behavior in the field, when taking all other factors into account. Finally, unlike commonly feared, we don't observe any indication that newer generations of DIMMs have worse error behavior.

Research Areas

Data Management

Meet the teams driving innovation

Our teams advance the state of the art through research, systems engineering, and collaboration across Google.

See our teams

Follow us

About Google

Google Products

Privacy

Terms

Help

Submit feedback

×

var scriptUrl = "https://www.gstatic.com/glue/v27_1/material-components-web.min.js";

var scriptElement = document.createElement('script');

scriptElement.async = false;

scriptElement.src = scriptUrl;

document.body.appendChild(scriptElement);

var scriptUrl = "https://www.youtube.com/player_api";

var scriptElement = document.createElement('script');

scriptElement.async = false;

scriptElement.src = scriptUrl;

document.body.appendChild(scriptElement);

var scriptUrl = "/gr/static/js/googleresearch.js?id=b23210a21057c86d701509e7dd1b5284";

var scriptElement = document.createElement('script');

scriptElement.async = false;

scriptElement.src = scriptUrl;

document.body.appendChild(scriptElement);

var scriptUrl = "https://support.google.com/inapp/api.js";

var scriptElement = document.createElement('script');

scriptElement.async = false;

scriptElement.src = scriptUrl;

document.body.appendChild(scriptElement);

var scripts = [

"https://www.gstatic.com/glue/cookienotificationbar/cookienotificationbar.min.js"

];

scripts.forEach(function(scriptUrl) {

var scriptElement = document.createElement('script');

scriptElement.async = false;

scriptElement.src = scriptUrl;

scriptElement.setAttribute("data-glue-cookie-notification-bar-category", "2B");

document.body.appendChild(scriptElement);

});
