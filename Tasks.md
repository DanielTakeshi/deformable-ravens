# Tasks

Here, I define the tasks from the vanilla Transporter Networks paper (presented
at CoRL 2020). The IDs for objects start at index 4 because earlier ones are
taken up by the starting plane and other things.

- [Insertion](#insertion)
- [Aligning](#aligning)
- [Cable](#cable)
- [Hanoi](#hanoi)
- [Sweeping](#sweeping)

## insertion

There's just one L-shaped block, hence:

```
steps = [{4: (angle,[0])}]
```

where ID=4 represents the L-shaped block. Strangely, the angle is just `2*pi`
which is 0 degrees, so I'm curious why it's there? The "0" in the list is
because that is our (only) candidate target, which is a key in placing.

The placing is:

```
places = {0: ((pos,orn))}
```

This makes sense. There's only one possible placing target, and we know its
position and orientation beforehand. Then, the oracle policy will only be able
to consider this as a placing point, and then it can take the position of this
target, and the position of the *object*, and then determine the translation
distance. The orientation is determined in a straightforward manner.

```
objects: [4]        # just the single block
fixed_objects: [5]  # just the single place hole
```


## aligning

Here we have a block and must align it with some corner. There is only one step
listed, which will look like this:


```
steps: [{5: (angle, [0, 1, 2, 3])}]
```

Again, angle is `2*np.pi`. But unlike insertion, here we have *four* possible
placing targets (since we only have a corner to align, not a full shape cavity
which limits rotations), each of which is pre-specified, such as with this info:

```
places = {
    0: ((0.340625, 0.03749, 0.0), (0.0, 0.0,  0.97137,  0.23755)),
    1: ((0.340625, 0.03749, 0.0), (0.0, 0.0,  0.23755, -0.97137)),
    2: ((0.344165, 0.02628, 0.0), (0.0, 0.0, -0.85484,  0.51889)),
    3: ((0.344165, 0.02628, 0.0), (0.0, 0.0,  0.51889,  0.85484))
}
```

The reason must be that to align a rectangular box with a corner, there are four
possible corners we can align it with. Hence, `places`, must take all four into
account. Then during the action, we consider all four places, and we find which
one of these is the closest to the picking pose.


## cable

This is somewhat unique in that we `self.metric = 'zone'` (and not 'pose') yet
this is another one where `self.primitive = pick_place'` because the agent does
pick and place actions with suctions (instead of sweeping).

The cable consists of 20 items attached to it. This results in 'steps' having 20
items:


```
steps: [{5: (0, [5]), 6: (0, [6]), 7: (0, [7]), 8: (0, [8]), 9: (0, [9]), 10: (0, [10]), 11: (0, [11]), 12: (0, [12]), 13: (0, [13]), 14: (0, [14]), 15: (0, [15]), 16: (0, [16]), 17: (0, [17]), 18: (0, [18]), 19: (0, [19]), 20: (0, [20]), 21: (0, [21]), 22: (0, [22]), 23: (0, [23]), 24: (0, [24])}]
```

Pretty simple: each index corresponds to some bead (in order), and each bead has
only one possible target, corresponding to its "place" location. When the oracle
takes an action, it looks at every single bead, and figures out which one is
furthest from its target location. Again, each bead (object) only has one
possible target, unlike (say) aligning where a single object could have multiple
valid target locations.

```
places: {
    5:  ((0.6798847, 0.057840355, 0.001), (0, 0, 0, 1.0)),
    6:  ((0.677263,  0.043943353, 0.001), (0, 0, 0, 1.0)),
    7:  ((0.6746414, 0.030046336, 0.001), (0, 0, 0, 1.0)),
    8:  ((0.6720198, 0.016149327, 0.001), (0, 0, 0, 1.0)),
    9:  ((0.6693981, 0.002252310, 0.001), (0, 0, 0, 1.0)),
    10: ((0.6667765, -0.011644706, 0.001), (0, 0, 0, 1.0)),
    11: ((0.6641548, -0.025541719, 0.001), (0, 0, 0, 1.0)),
    12: ((0.6615332, -0.039438732, 0.001), (0, 0, 0, 1.0)), 13: ((0.6589115, -0.053335745, 0.001), (0, 0, 0, 1.0)), 14: ((0.6562899, -0.06723276, 0.001), (0, 0, 0, 1.0)), 15: ((0.6536682, -0.081129774, 0.001), (0, 0, 0, 1.0)), 16: ((0.6510466, -0.09502679, 0.001), (0, 0, 0, 1.0)), 17: ((0.648425, -0.10892381, 0.001), (0, 0, 0, 1.0)), 18: ((0.64580333, -0.12282081, 0.001), (0, 0, 0, 1.0)), 19: ((0.6431817, -0.13671783, 0.001), (0, 0, 0, 1.0)), 20: ((0.64056003, -0.15061484, 0.001), (0, 0, 0, 1.0)), 21: ((0.6379384, -0.16451186, 0.001), (0, 0, 0, 1.0)), 22: ((0.6353167, -0.17840886, 0.001), (0, 0, 0, 1.0)), 23: ((0.6326951, -0.1923059, 0.001), (0, 0, 0, 1.0)),  24: ((0.6300734, -0.2062029, 0.001), (0, 0, 0, 1.0))
}
```

Looks straightforward: the 'places' specifies a line, which is why the x and y
above change in a linear fashion, and the z and rotations are fixed (the beads
are rotationally symmetric so orientation doesn't matter). The line is
specified by the missing edge in the green square (of which three edges exist
at the start).

One item in `self.object_points` for each bead in the cable. Then in the
zone-based reward function computation, we iterate through each bead to check if
it is inside the acceptable zone, and simply compute the fraction of beads in
the zone (with a delta over the prior time step to get the final reward).

What about the template? It uses these:

```
ravens/assets/square/square-template.urdf
ravens/assets/line/line-template.urdf
```

The name is misleading, though, because the square only has three out of four
lines visible. (If you look at the .urdf, you'll see three `<visual></visual>`
blocks corresponding to the three green edges.) Then besides the square
template, there is a line template which should be the target for the cable. The
line is actually NOT added (commented out) in Andy's code, but it's useful to
visualize.

## hanoi

In hanoi, it's a bit more complicated, and looks like this at the start:

```
steps: [{5: (0, [0])}, {6: (0, [1])}, {5: (0, [2])}, {7: (0, [3])}, {5: (0, [4])}, {6: (0, [5])}, {5: (0, [6])}]
```

In hanoi, we solve the sequence with dynamic programming. Since it's suctioning,
we don't do rotation, hence the (angle,[...]) turns to (0,[...]) at each step.

## sweeping

A unique environment where we use a spatula to sweep 50 "morsels" into a zone.
Each of those 50 items goes in `self.object_points` and we compute the fraction
that are within the zone area for the reward.

Be careful about `self.zone_size = (0.12, 0.12, 0)`!! That is used in the reward
function computation because it "re-maps" all the morsels (via the INVERSE of
the zone pose) back to the base link, and then computes the number of points
within the `self.zone_size` boundary. The reason why it's set to these values is
that the `zone.obj` ranges from -10 to 10 in each coordinate, hence the range of
x and y is 20, and 20 times 0.006 (the scale in `zone.urdf` for x and y) is
0.12. Ah!
