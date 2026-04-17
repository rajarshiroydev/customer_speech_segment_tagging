# YOYO Conversation Boundary Detection

This project explores multiple ways to detect the start and end timestamps of customer conversations in noisy retail-shop audio. The input contains three audio files, each with two customer interactions. The recordings include English/Kannada/Hindi speech, background shop noise, silence, staff-side conversations, and long pauses inside real customer conversations.

The practical goal is not just to detect speech. The system needs to identify the portions of speech that correspond to customer-facing conversations and return two final conversation intervals per audio file.

## Project Structure

```text
audio/                 Input audio files
approaches/            Five experiment notebooks and shared output utilities
evaluations/           Lightning AI evaluation CSVs for each approach
ground_truth.json      Reference conversation boundaries
problem_statement.md   Assignment prompt
context.md             Development notes and approach rationale
```

Each approach notebook writes standardized outputs so the approaches can be compared consistently:

```text
*_conversation_candidates.csv          Candidate speech/conversation blocks
*_final_two_conversations.csv          Final two selected customer conversations
*_tagged_conversation_candidates.csv   Candidates annotated with selected customer assignment
evaluation_against_ground_truth.csv    Per-conversation metric comparison
evaluation_summary.csv                 Per-audio aggregated metrics
```

## Evaluation Metrics

The evaluation compares predicted conversation boundaries against the reference annotations in `ground_truth.json`.

| Metric | Meaning |
|---|---|
| `abs_start_error_s` | Absolute error between predicted and reference start time |
| `abs_end_error_s` | Absolute error between predicted and reference end time |
| `mean_boundary_mae_s` | Average of all start and end absolute errors |
| `IoU` | Temporal intersection-over-union between predicted and reference interval |

Higher `IoU` is better. Lower boundary error is better. I used mean IoU as the main ranking metric because the task is interval detection, not only point-boundary detection.

## Approaches

| Approach | Method | Pros | Cons |
|---|---|---|---|
| Approach 1: Energy Baseline | Uses frame-level RMS energy to find active audio regions. A relative threshold (15% of the file's 95th-percentile RMS) makes detection volume-adaptive. Merges nearby active regions and selects the top-2 longest candidates. No neural model required. | Extremely fast, zero model dependencies, fully interpretable (energy curve is directly visible), useful as a true signal-processing floor. | Cannot distinguish speech from other loud sounds — shop noise, clanging metal, or music all cross the energy threshold equally. Sensitive to volume variation across files. No understanding of speech patterns, speaker identity, or semantics. |
| Approach 2: VAD Only | Uses Silero VAD to detect speech segments, merges them into conversation candidates, and selects final intervals. | More robust than raw energy — neural model trained specifically for speech, not just loudness. Multilingual speech detection works reasonably well, still lightweight and explainable. | Detects all speech including staff chatter; long pauses split real conversations; no speaker or semantic understanding. |
| Approach 3: VAD + Diarization | Uses VAD candidates and pyannote speaker diarization to inspect speaker turns and multi-speaker structure. | Adds speaker-turn information; theoretically useful for identifying customer-salesperson interactions. | Diarization is unstable in noisy retail audio; speaker labels do not reliably persist across distant regions; extra dependency and Hugging Face model access complexity. |
| Approach 4: WhisperX + Diarization | Uses WhisperX ASR/alignment plus diarization and candidate scoring to select customer conversations. | Adds transcript and timing information; performs much better on English audio; gives useful debug artifacts for downstream logic. | Kannada/mixed-language transcription is weaker; still struggles with long pauses and non-customer speech; heavier runtime and dependency setup. |
| Approach 5: Hybrid Post-Processing | Uses Approach 4 artifacts with quality-aware rules: VAD structure, ASR density, long-pause handling, transcript quality checks, and exactly-two-conversation prior. | Best overall results; handles long pauses better; robust fallback when ASR quality is low; most aligned with the real assignment objective. | More hand-engineered; depends on Approach 4 artifacts; can still fail when the true split is ambiguous or ASR/VAD artifacts are poor. |

## Evaluation Results

The following scores are compiled from the Lightning AI runs in `evaluations/`.

| Rank | Approach | Mean IoU | Mean Start Error (s) | Mean End Error (s) | Mean Boundary MAE (s) |
|---:|---|---:|---:|---:|---:|
| 1 | Approach 5: Hybrid | 0.778 | 68.988 | 23.319 | 46.153 |
| 2 | Approach 4: WhisperX | 0.377 | 75.500 | 96.399 | 85.949 |
| 3 | Approach 1: Energy Baseline | 0.372 | 104.853 | 208.088 | 156.471 |
| 4 | Approach 2: VAD Only | 0.250 | 118.969 | 184.468 | 151.719 |
| 5 | Approach 3: VAD + Diarization | 0.233 | 75.004 | 216.631 | 145.817 |

### Per-file breakdown — Approach 1 (Energy Baseline)

| Audio file | Mean IoU | Mean Start Error (s) | Mean End Error (s) |
|---|---:|---:|---:|
| Sample1KN.mp3 | 0.697 | 107.160 | 0.685 |
| Sample2EN.mp3 | 0.118 | 207.200 | 498.470 |
| sample3KN.mp3 | 0.303 | 0.200 | 125.110 |

The Kannada file (Sample1KN) scores surprisingly well on end-boundary detection (mean end error < 1 s) because the energy envelope of that file happens to align closely with the true conversation spans. The English file (Sample2EN) scores very poorly because dense background shop noise inflates the energy throughout the file, causing the threshold to activate on non-speech regions and miss the second conversation entirely.

## Why Approach 5 Wins

Approach 5 wins because it treats the task as conversation-boundary detection rather than simple speech detection. It uses the stronger signals from Approach 4, but adds post-processing that reflects the structure of the problem:

- each file is expected to contain exactly two customer conversations;
- a long pause does not always mean a new customer conversation;
- Kannada/mixed-language ASR can be unreliable, so transcript confidence must influence the logic;
- dense ASR activity and VAD structure are useful for estimating broad conversation spans;
- final intervals should be selected globally, not just by choosing the longest isolated speech chunks.

This is especially visible on `sample3KN.mp3`, where Approach 5 reaches a mean IoU of about `0.981`, and on `Sample2EN.mp3`, where WhisperX-based transcript structure helps improve the customer conversation spans.

## Why The Other Approaches Score Lower

**Approach 1** (Energy Baseline) detects active audio regions but has no concept of speech. In a retail shop environment with constant background noise, the energy threshold fires on non-speech sounds as readily as on speech. It scores higher than Approach 2 overall due to a favourable result on Sample1KN, but fails badly on the English file where dense ambient noise masks the true conversation structure.

**Approach 2** (VAD Only) replaces the energy threshold with Silero VAD, a neural model that specifically recognises human speech patterns. This makes it more robust to non-speech noise than Approach 1, but it still detects all speech equally — customer, staff, and incidental talk. The top-2 longest selection heuristic then fails when non-customer speech is the dominant signal.

**Approach 3** adds diarization, but diarization alone does not solve the task. It identifies speaker clusters, not customer identities. In noisy multilingual retail audio, speaker labels can be fragmented or inconsistent, which caused poor end-boundary estimates and ranked it last overall.

**Approach 4** improves substantially on English audio because ASR gives semantic and timing information. However, it still underperforms on Kannada/mixed-language samples and can select only part of a longer conversation when there are long pauses or weak transcription.

**Approach 5** keeps the useful ASR/VAD signals from Approach 4 but adds task-specific correction rules. This makes it the best final candidate among the tested methods.

## Future Improvements

The next improvements would be to replace rule-based post-processing with a supervised boundary classifier, improve multilingual ASR for Kannada/Hindi-English code-switching, and use more labeled audio to tune conversation-level decision rules.