# Streaming Fraud Analysis
Evan Frangipane

## 🧠 Goal

This project is a sandbox for running ML on streaming fraud (bots). I
could not find a decent dataset for streaming data as Spotify (🤮)
scrubbed the internet of the MSSD. The next best option was to create my
own streaming data. This gave me a ton of customizability, but it’s
difficult to say how well this matches up to data from real users. With
my dataset filled with baseline users I can add in the bots and see what
techniques are best at picking them out.

------------------------------------------------------------------------

## 🤹 Baseline Data

I defined many parameters to help me create the dataset including number
of users, number of artists, number of tracks per artists (avg), start
and end dates, artist and track popularity params, track length params,
etc. Also, I implemented a weekend boost to streams.

Here is a snippet of the track dataframe:

<table>
<colgroup>
<col style="width: 10%" />
<col style="width: 18%" />
<col style="width: 14%" />
<col style="width: 17%" />
<col style="width: 16%" />
<col style="width: 22%" />
</colgroup>
<thead>
<tr>
<th>artist_id</th>
<th>track_id</th>
<th>track_len_sec</th>
<th>artist_popularity</th>
<th>track_popularity</th>
<th>total_track_popularity</th>
</tr>
</thead>
<tbody>
<tr>
<td>artist_1</td>
<td>artist_1_track_1</td>
<td>234.836</td>
<td>0.104328</td>
<td>0.097664</td>
<td>0.010189</td>
</tr>
<tr>
<td>artist_1</td>
<td>artist_1_track_2</td>
<td>179.183</td>
<td>0.104328</td>
<td>0.072737</td>
<td>0.007589</td>
</tr>
<tr>
<td>artist_1</td>
<td>artist_1_track_3</td>
<td>237.418</td>
<td>0.104328</td>
<td>0.061220</td>
<td>0.006387</td>
</tr>
<tr>
<td>artist_1</td>
<td>artist_1_track_4</td>
<td>274.281</td>
<td>0.104328</td>
<td>0.054172</td>
<td>0.005652</td>
</tr>
<tr>
<td>artist_1</td>
<td>artist_1_track_5</td>
<td>188.202</td>
<td>0.104328</td>
<td>0.049269</td>
<td>0.005140</td>
</tr>
</tbody>
</table>

Next I defined some parameters for individual users like do they like
single artists sessions or mixed, more popular music, avg songs per
session, avg sessions per day, song skipping probability.

Here is a snippet of the user dataframe:

<table>
<thead>
<tr>
<th>user_id</th>
<th>mix_sesh</th>
<th>pop_fan</th>
<th>sesh_len</th>
<th>sesh_num</th>
<th>skip_prob</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>0.770236</td>
<td>0.449934</td>
<td>7</td>
<td>1</td>
<td>0.722269</td>
</tr>
<tr>
<td>2</td>
<td>0.678292</td>
<td>0.198552</td>
<td>7</td>
<td>2</td>
<td>0.449958</td>
</tr>
<tr>
<td>3</td>
<td>0.631687</td>
<td>0.335670</td>
<td>2</td>
<td>1</td>
<td>0.155921</td>
</tr>
<tr>
<td>4</td>
<td>0.583200</td>
<td>0.587551</td>
<td>8</td>
<td>2</td>
<td>0.538524</td>
</tr>
<tr>
<td>5</td>
<td>0.622390</td>
<td>0.508089</td>
<td>5</td>
<td>1</td>
<td>0.514179</td>
</tr>
</tbody>
</table>

From here I generated the stream data. I looped over each user and
rolled dice for streams over the streaming period.

Here is a snippet of the streams dataframe:

<table>
<colgroup>
<col style="width: 6%" />
<col style="width: 9%" />
<col style="width: 21%" />
<col style="width: 8%" />
<col style="width: 17%" />
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 6%" />
</colgroup>
<thead>
<tr>
<th>user_id</th>
<th>session_id</th>
<th>timestamp</th>
<th>artist_id</th>
<th>track_id</th>
<th>track_duration_sec</th>
<th>listen_duration_sec</th>
<th>is_bot</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>0</td>
<td>2025-01-02 21:41:31.000000000</td>
<td>artist_30</td>
<td>artist_30_track_30</td>
<td>172.992492</td>
<td>54.357032</td>
<td>False</td>
</tr>
<tr>
<td>1</td>
<td>0</td>
<td>2025-01-02 21:42:25.357031881</td>
<td>artist_30</td>
<td>artist_30_track_10</td>
<td>205.152908</td>
<td>205.152908</td>
<td>False</td>
</tr>
<tr>
<td>1</td>
<td>0</td>
<td>2025-01-02 21:45:50.509940231</td>
<td>artist_30</td>
<td>artist_30_track_25</td>
<td>209.933866</td>
<td>96.338037</td>
<td>False</td>
</tr>
<tr>
<td>1</td>
<td>0</td>
<td>2025-01-02 21:47:26.847977059</td>
<td>artist_30</td>
<td>artist_30_track_9</td>
<td>196.974013</td>
<td>35.726356</td>
<td>False</td>
</tr>
<tr>
<td>1</td>
<td>1</td>
<td>2025-01-03 06:48:22.000000000</td>
<td>artist_8</td>
<td>artist_8_track_9</td>
<td>89.516270</td>
<td>39.074505</td>
<td>False</td>
</tr>
</tbody>
</table>

------------------------------------------------------------------------

## 🤖 Botting data

I want to create a variety of bots with different behavior to see what
we are sensitive to. The first type of botting I implemented is long
duration stream bots for single artists. One can imagine purchasing N
streams for an artist over a time period. I create n bots that stream
for some long duration (like 5 hours) then cool off for some waiting
period (like 12 hours). The N total streams are divided among the bots
and they are staggered throughout the time period but they still have
regular patterns.

Searching for these bots will involve looking for repeating patterns
that would be suspicious of a human user. To improve the bot I want to
create some human-like baseline data for the bot and then on top of that
have long duration streams to try and mask the bottiness. This will
require the analyst to come up with more clever methods of detection.

------------------------------------------------------------------------

## TODO

Peakiness of streams -\> high #s for weeks, month? -\> bot? User
variations -\> users who increase, decrease listening Fix the bot number
of sessions scaling to work more as expected
