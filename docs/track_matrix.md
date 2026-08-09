# Track Matrix

| Track | Domain | Main mechanism | Required baseline | Primary diagnostic |
|---|---|---|---|---|
| KAN | General | learned edge functions | MLP | symbolic/function approximation |
| xLSTM | Sequences | exponential gates + scalar/matrix memory | LSTM + Transformer | recall/state tracking |
| Mamba-3 | Sequences | selective/state-space recurrence | Transformer + GRU/xLSTM | state tracking + long sequence throughput |
| TTT | Sequences | learner as hidden state | Transformer + recurrent baseline | associative recall under context growth |
| Titans | Sequences/memory | neural long-term memory | Transformer + TTT/SSM | long-context retrieval |
| Hope | Continual/sequence | nested learning timescales | TTT/Titans-style baseline | continual incorporation/forgetting |
| PFN/TabPFN | Tabular | in-context prediction over task priors | CatBoost/GBDT + MLP | synthetic prior recovery |
| Relational FM | Relational data | native multi-table representation | flatten-then-GBDT/GNN | FK/temporal relational task |
| Sparse MoE | General/sequence | conditional expert routing | dense FFN | routing balance/specialization |
| Flow Matching | Generative | vector-field regression | simple diffusion/flow baseline | 2D density recovery |
| JEPA | Representation/world model | predict latent targets | autoencoder/contrastive baseline | latent-factor prediction |
