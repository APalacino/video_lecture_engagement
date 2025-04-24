# Video lecture engagement

#### -- Project Status: In progress

## Project Intro/Objective
The purpose of this project is to choose the best model to predict if a video will be engaging to a user not, using the data from [VLE data set](https://github.com/sahanbull/VLE-Dataset).


### Methods Used
* Machine Learning
* Data Visualization
* Predictive Modeling

### Technologies
* Python
* Pandas, jupyter
* Scikit-learn 
* Matplotlib, seaborn

## Project Description
Use the Video Lecture Engagement to predict if a video will be engaging to a user. This data set is the result of a previously processed data from an open source. The objective is to choose the best model from at least three optimized machine learning algorithms, they will be optimized for Fbeta prioritizing precision over recall.

From the repository of the data set the feature descriptions are the following:

<table class="tg">
<thead>
  <tr>
    <th class="tg-fymr">Variable Type</th>
    <th class="tg-fymr">Name</th>
    <th class="tg-fymr">Quality Vertical </th>
    <th class="tg-fymr">Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8bgf" colspan="4">Metadata-based&nbsp;&nbsp;&nbsp;Features</td>
  </tr>
  <tr>
    <td class="tg-0pky">cat.</td>
    <td class="tg-0pky">Language</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">Language of instruction of the video lecture</td>
  </tr>
  <tr>
    <td class="tg-0pky">cat.</td>
    <td class="tg-0pky">Domain</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">Subject area (STEM or Miscellaneous)</td>
  </tr>
  <tr>
    <td class="tg-8bgf" colspan="4">Content-based&nbsp;&nbsp;&nbsp;Features</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Word Count</td>
    <td class="tg-0pky">Topic Coverage </td>
    <td class="tg-0pky">Word Count of Transcript</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Title Word Count</td>
    <td class="tg-0pky">Topic Coverag </td>
    <td class="tg-0pky">Word Count of Title</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Document Entropy</td>
    <td class="tg-0pky">Topic Coverage </td>
    <td class="tg-0pky">Document Entropy of Transcript</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Easiness (FK Easiness)</td>
    <td class="tg-0pky">Understandability  </td>
    <td class="tg-0pky">FK Easiness based on FK Easiness </td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Stop-word Presence Rate</td>
    <td class="tg-0pky">Understandability</td>
    <td class="tg-0pky">Stopword Presence Rate of Transcript text</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Stop-word Coverage Rate</td>
    <td class="tg-0pky">Understandability </td>
    <td class="tg-0pky">Stopword Coverage Rate of Transcript text</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Preposition Rate</td>
    <td class="tg-0pky">Presentation </td>
    <td class="tg-0pky">Preposition Rate of Transcript text</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Auxiliary Rate</td>
    <td class="tg-0pky">Presentation </td>
    <td class="tg-0pky">Auxiliary Rate of Transcript text</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">To Be Rate</td>
    <td class="tg-0pky">Presentation </td>
    <td class="tg-0pky">To-Be Verb Rate of Transcript text</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Conjunction Rate</td>
    <td class="tg-0pky">Presentation </td>
    <td class="tg-0pky">Conjunction Rate of Transcript text</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Normalisation Rate</td>
    <td class="tg-0pky">Presentation </td>
    <td class="tg-0pky">Normalisation Rate of Transcript text</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Pronoun Rate</td>
    <td class="tg-0pky">Presentation </td>
    <td class="tg-0pky">Pronoun Rate of Transcript text</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Published Date</td>
    <td class="tg-0pky">Freshness </td>
    <td class="tg-0pky">Duration between 01/01/1970 and the lecture published date (in days)</td>
  </tr>
  <tr>
    <td class="tg-8bgf" colspan="4">Wikipedia-based&nbsp;&nbsp;&nbsp;Features</td>
  </tr>
  <tr>
    <td class="tg-0pky">cat.</td>
    <td class="tg-0pky">Top-5 Authoritative Topic URLs</td>
    <td class="tg-0pky">Authority </td>
    <td class="tg-0pky">5 Most Authoritative Topic URLs based on PageRank Score. 5 features in&nbsp;&nbsp;&nbsp;this group</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Top-5 PageRank Scores </td>
    <td class="tg-0pky">Authority </td>
    <td class="tg-0pky">PageRank Scores of the top-5 most authoritative topics</td>
  </tr>
  <tr>
    <td class="tg-0pky">cat.</td>
    <td class="tg-0pky">Top-5 Covered Topic URLs</td>
    <td class="tg-0pky">Topic Coverage </td>
    <td class="tg-0pky">5 Most Covered Topic URLs based on Cosine Similarity Score. 5 features in&nbsp;&nbsp;&nbsp;this group</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Top-5 Cosine Similarities</td>
    <td class="tg-0pky">Topic Coverage   </td>
    <td class="tg-0pky">Cosine Similarity Scores of the top-5 most covered topics</td>
  </tr>
  <tr>
    <td class="tg-8bgf" colspan="4">Video-based Features</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Lecture Duration</td>
    <td class="tg-0pky">Topic Coverage </td>
    <td class="tg-0pky">Duration of the video (in seconds)</td>
  </tr>
  <tr>
    <td class="tg-0pky">cat.</td>
    <td class="tg-0pky">Is Chunked</td>
    <td class="tg-0pky">Presentation </td>
    <td class="tg-0pky">If the lecture consists of multiple videos</td>
  </tr>
  <tr>
    <td class="tg-0pky">cat.</td>
    <td class="tg-0pky">Lecture Type</td>
    <td class="tg-0pky">Presentation </td>
    <td class="tg-0pky">Type of lecture (lecture, tutorial, invited talk etc.)</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Speaker speed</td>
    <td class="tg-0pky">Presentation </td>
    <td class="tg-0pky">Speaker speed (words per minute)</td>
  </tr>
  <tr>
    <td class="tg-0pky">con.</td>
    <td class="tg-0pky">Silence Period Rate (SPR)</td>
    <td class="tg-0pky">Presentation</td>
    <td class="tg-0pky">Fraction of silence in the lecture video</td>
  </tr>
</tbody>
</table>


## Needs of this project

- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.

    *If using offline data mention that and how they may obtain the data from the froup)*
    
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructions](Link to file)

## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](link)
* [Notebook/Markdown/Slide DeckTitle](link)
* [Blog Post](link)

## Contact
* e-mail: appalacinoc@gmail.com
* LinkedIn: https://www.linkedin.com/in/angela-palacino/
