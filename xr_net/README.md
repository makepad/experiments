How to run xr-net example:

```
git clone makepad; git checkout dev
git clone experiments
```

> you would have to put both dirs at the same level (since I do ../makepad/widgets)

```
makepad/
experiments/
```

and then you build for android using

```
cd makepad
cargo install --path=./tools/cargo_makepad
cargo makepad android install-toolchain
cd ../experiments
cargo makepad android --variant=quest run -p makepad-experiment-xr-net --release
```

if you have your quest in 'dev' mode then it should run

one final thing you need to do:
go to `settings` in your quest > `privacy and safety` > `installed apps` > xr-net > `spatial data`
and set the checkbox for that thing
