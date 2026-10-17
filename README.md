# gemcraft
integration of solidcraft with gemma

gemcraft takes in user text and image input, and makes 3d cad models in [solidcraft](https://github.com/tangency-labs/solidcraft) through an ai model

solidcraft-cli binary needed in the root folder
clone the solidcraft git repo

```
git clone https://github.com/tangency-labs/solidcraft.git && cd solidcraft
cargo build --release -p solidcraft-cli
```
move the binary at /target/release/solidcraft-cli to the root of this repo


add a .env file with :
```
GEMINI_API_KEY=<your api key>
```
or point your local ai server in the code

run the command:
```
CAD_MODEL=gemma-4-26b-a4b-it CAD_FALLBACKS=gemini-3.5-flash-lite CAD_KNOWLEDGE=examples.md python cad_agent.py "<your prompt goes here>" <image.png (optional)>
```
