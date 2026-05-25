# DevLog-001: Shift Register Deserializer Visualization Proposal

## Goal

Add a deep-dive visualization that explains how sampled serial bits are stored into a 16-bit word. This should be more concrete than an abstract "serial-in parallel-out" block, but still higher level than transistor schematics. The user should be able to see how each UI sample picks a storage basket, how the bit is held, and how the final output latch makes all 16 lanes valid together.

Core idea:

```text
recovered data sample
  -> retiming latch
  -> bit-position counter
  -> storage basket / shift cell
  -> full 16-bit assembly register
  -> parallel output latch
  -> stable core-facing bus
```

## Educational Questions To Answer

- When a serial bit arrives, how does the receiver decide which bit position gets it?
- Is the internal shift register staggered in time?
- Why do the final parallel output lanes update together?
- What is the difference between shift-register cells and output-latch lanes?
- What does the divide-by-16 word clock do?
- Where does one word end and the next word begin?

## Visual Model Level

Use a clocked-register model:

- Each storage cell is a flip-flop/latch-level block with input, stored value, and output.
- A modulo-16 counter points to the current bit position.
- The counter output is decoded into one-hot write enables, e.g. `counter=0011` asserts only `WE[3]`.
- The sampled data is usually distributed on a data bus or through staged demux paths; the selected write enable opens a clocked mux/transmission-gate path or enables a D flip-flop/latch input.
- A word-valid pulse captures the assembled word into an output latch.
- Internal cells may update one UI at a time.
- Output bus updates all 16 bits at one word boundary.

Do not include:

- transistor-level flip-flop schematic
- setup/hold violation math
- metastability analysis
- protocol framing details beyond a simplified "known word boundary"

## Two Storage Interpretations To Show

### 1. True Shift Chain

Classic serial-in parallel-out shift register:

```text
new_bit -> cell0 -> cell1 -> cell2 -> ... -> cell15
```

At every UI:

```text
cell15 <= cell14
cell14 <= cell13
...
cell1  <= cell0
cell0  <= new_bit
```

This answers "how bits move through storage cells."

Important visual point:

- The arriving bit does not choose a named `b7` basket directly.
- Instead, bits shift through a chain.
- After 16 shifts, the full word appears across the register.
- Depending on shift direction, the physical cell order may be reversed relative to `b0...b15`.

### 2. Basket / Demuxed Register Bank

Addressed storage view:

```text
bit_position_counter = 0..15
incoming_sample -> selected basket b[counter]
```

At every UI:

```text
if counter == 0: basket b0 captures sample
if counter == 1: basket b1 captures sample
...
if counter == 15: basket b15 captures sample
```

This answers "how the bit picks the basket."

Implementation note:

- Real designs may use shift chains, staged demuxes, interleaved banks, or hybrids.
- For teaching, the basket model is the clearest way to explain indexed storage.
- The staged demux tree already on the page is the hardware-flavored version of basket selection.
- "Selecting basket 3" is not a single ideal switch in real CMOS. A simple low-speed register-bank view is: counter bits feed a 4-to-16 decoder, `WE[3]` drives a local clock-enable or transmission gate, and the sampled data is captured by the b3 latch/DFF. A high-speed SerDes may implement the same routing idea with a hierarchy of latches and demux stages rather than a large one-hot bus.

## Main Panels

### 1. Serial Sample And Bit Counter

Show:

```text
sampled serial bit
recovered UI clock
modulo-16 counter
current bit index
```

Animation:

- Counter advances `0,1,2,...,15,0`.
- The active counter value glows.
- A token representing the sampled bit moves toward the selected storage cell.

Waveforms:

```text
sampled_data
UI_sample_clock
counter[3:0]
word_valid
```

### 2. Basket Selection Register Bank

Show 16 storage baskets:

```text
b0 b1 b2 b3 ... b15
```

Each basket displays:

```text
bit label
stored 0/1 value
last update time
input enable glow
small local waveform
```

Animation:

- Current sampled bit leaves the sampler.
- The modulo-16 counter highlights one basket.
- Only that basket's write-enable wire glows.
- A detailed slice, such as `b3`, shows `counter[3:0] -> decoder -> WE[3] -> clocked mux/transmission gate -> latch/DFF`.
- The selected basket captures the bit.
- Previously filled baskets remain stable.

This directly answers: "how does the shift register pick the basket and store the bit?"

### 3. True Shift Chain View

Show 16 cells in a row with arrows:

```text
sample -> [0] -> [1] -> [2] -> ... -> [15]
```

Animation:

- On each UI tick, all arrows flash.
- Values move one cell to the right.
- The new sample enters cell 0.
- A small history marker shows how a bit sampled earlier moves through the chain.

This is a separate view or toggle from basket selection, because both on screen at once may be too dense.

### 4. Output Latch And Timing Reset

Show two rows:

```text
assembly register: cells fill/update at UI cadence
output latch: all 16 lanes update together at word_valid
```

Animation:

- Assembly register updates one bit per UI.
- At counter wrap / word boundary, `word_valid` flashes.
- All output latch cells copy the completed word simultaneously.
- Output bus remains stable for one 143 ps word period.

Label clearly:

```text
internal assembly timing is staggered
core-facing output timing is aligned
```

### 5. Per-Lane Waveform Panel

Show all 16 output lanes with tiny traces:

```text
assembly b0 waveform
assembly b1 waveform
...
assembly b15 waveform
```

and separately:

```text
latched output b0 waveform
latched output b1 waveform
...
latched output b15 waveform
```

Key distinction:

- Assembly traces update at different UI offsets.
- Output latch traces update together on word boundaries.

## Suggested Interaction Controls

- `Storage model`: basket bank / true shift chain / both.
- `Follow bit`: highlight one sampled bit as it moves.
- `Show output latch`: toggle final synchronized bus.
- `Show lane waveforms`: toggle all 16 mini waveforms.
- `Word boundary`: known boundary / shifted boundary demo.
- `Step one UI`: pause and advance exactly one sample.

## Data Model Sketch

State variables:

```text
sampleIndex       // total UI samples received
bitIndex          // sampleIndex % 16
wordIndex         // floor(sampleIndex / 16)
assembly[16]      // partially filled current word
assemblyValid[16] // which baskets have been written this word
outputLatch[16]   // last complete word visible to core
trackedBit        // optional token for one sampled bit
```

Per UI sample:

```text
1. sample = serialData[wordIndex][bitIndex]
2. assembly[bitIndex] = sample
3. assemblyValid[bitIndex] = true
4. if bitIndex == 15:
     outputLatch = assembly
     pulse word_valid
     clear assemblyValid for next word
5. bitIndex advances
```

For true shift-chain mode:

```text
1. cell15 <= cell14
2. cell14 <= cell13
3. ...
4. cell0 <= sample
5. outputLatch captures mapped cells at word boundary
```

## Accuracy Notes

- The output latch adds word-level latency. `b0` waits almost a full word period before appearing on the parallel output, while `b15` waits much less.
- The final output latch "resets" timing for the digital core: all 16 output lanes become valid together.
- Word alignment is assumed known for the first version. A future view can add a training marker or comma-alignment concept.
- This view should connect to the demux tree: staged demuxing is one hardware way to distribute sampled bits into slower lanes.

## Implementation Feedback

- Implemented in `serdes_cdr_shift_register_deep_dive.html` as basket-bank, shift-chain, and master/slave timing panels.
- Expanded basket selection from one example slice to all 16 baskets: shared data bus, 4-to-16 decoder, one-hot `WE[i]`, local `TG/mux`, latch, and stored value.
- Added shift-mechanism deep dive with DFF stages split into master/slave latches, clock phase, snapshot equations, setup/hold aperture, and `tCQ` update timing.
- Useful next improvement: add a toggle between addressed basket-bank storage and pure cascaded DFF shifting, with the same tracked bit shown in both.
