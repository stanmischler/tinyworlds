# Judging guide: what did Link do between two game frames

Each sheet holds 4 numbered panels (#id at top left of each row) from a top-down Zelda game. Link is the small player sprite, usually in a green tunic and cap. Judge each panel on its own; panels on one sheet are unrelated in time.

## Panel layout, left to right
1. **frame t**.
2. **frame t+1** (the later frame).
3. **frame t with the camera shift removed**: the scenery should line up with t+1. Grey = no source pixels.
4. **zoom of aligned t**.
5. **zoom of t+1**: the same region as 4. The region is an automatic guess of Link's position and may be wrong. If so, use the full frames (3 and 2) instead.

Yellow edge ticks sit in the same place in every frame, so positions can be compared: every 16 game pixels on panels 1 to 3, every 4 game pixels on the zooms.

**Compare the aligned frame (3, 4) with frame t+1 (2, 5).** Any displacement of Link there is his movement in the world, the button pressed. Link's on-screen motion in the first two frames does not matter: when the camera follows him he may stay in the same screen position while walking.

## Check the alignment first
**alignment_ok**: false if the scenery does not line up between aligned t and t+1, even by a clear offset. Moving characters do not count as scenery. Some panels are deliberately misaligned. When alignment_ok is false, label UNCLEAR with reason BAD_ALIGNMENT.

## Labels: one JSON object per panel

**label**: exactly one of
- **U, D, L, R, UL, UR, DL, DR**: Link's body is displaced in the aligned view, in screen directions (U = up the screen).
  - A move is at least 3 game pixels (about 15 screen pixels on the zooms). Less than that is not a move.
  - A diagonal needs at least 3 game pixels on BOTH axes. Otherwise use the single direction.
  - Judge the body or feet, not a raised arm or an item.
- **STANDING**: same place, same pose apart from small flicker. Walk-cycle leg motion with no body displacement is STANDING.
- **TURN**: same place, facing changes.
- **ATTACK**: sword swing with no move of at least 3 game pixels. If Link moves and swings at once, the move wins: label the direction.
- **INTERACT**: lifting, throwing or opening, in place. If he moves while doing this, label the direction.
- **UNCLEAR**: no action can be read. Give the reason.

**unclear_reason**: only when the label is UNCLEAR, else null. One of TEXT (dialog or text box open in either frame, whatever Link does), FADE_OR_CUT, NO_LINK (Link not visible in either frame), MENU, BAD_ALIGNMENT.

**confidence**: "clear" or "unsure". Still give your best label when unsure.

## Worked examples (guide_example.png)
- **#1055 = DR.** The camera moves; grass and cliffs line up between panels 3 and 2. Link's body is lower and further right in t+1, by clear amounts on both axes.
- **#674 = UR.** The camera moves; the blue arrows line up. Link's feet are higher by about 10 game pixels and further right by about 5. That he also changes facing does not matter, because he is displaced.
- **#946 = L.** The camera is still. Link's facing changes, but his body also ends up clearly further left, so this is a move, not a TURN.
- **#23 = TURN.** The camera is still. Link stays in the same place and changes from facing left to facing away.
- **#1697 = STANDING.** The camera is still, and Link is in the same place with the same pose.
- **#168 = ATTACK.** The camera is still, Link stays in place, and a sword arc appears beside him.
- **#964 = UNCLEAR, reason TEXT.** A text box is open.

For all of these, alignment_ok is true.
