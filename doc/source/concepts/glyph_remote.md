# Glyph remote

glyph-remote is shipped together with the glyph package. After installation, the `glyph-remote` command is available at the command line.

## Concept

With glyph-remote the separation between optimization method and optimization task is made easy. glyph-remote runs multi IO symbolic regression and sends candidate solution via ZeroMQ to an experiment controller for assessment. Every hyper-parameter used is assessable and fully configurable.

## Configuration

For a full list of configuration options type `glyph-remote --help`.

All hyper-parameters and algorithms used have default values.
However, it is mandatory to provide a information about the primitives you want to use:

key: primitives

value: mapping function name:arity.
