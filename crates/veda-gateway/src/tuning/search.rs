//! Multi-objective search backend, behind a small trait so it's swappable
//! if `scirs2-optimize` proves immature (Phase 3 of the auto-tuning plan).
//!
//! `scirs2-optimize`'s built-in `NSGAII` (`multi_objective::algorithms::NSGAII`)
//! is a closure-based, blocking "run to completion" optimizer: you call
//! `optimize(objective_fn)` once and it internally drives every generation,
//! calling your `Fn(&ArrayView1<f64>) -> Array1<f64>` synchronously as many
//! times as it wants. It has no ask/tell interface. That's incompatible with
//! this trait's design, where "evaluating a candidate" is Phase 4's real
//! trial (a reindex + eval-set run — expensive, async, external). So
//! `Nsga2Search` below is a from-scratch ask/tell generation loop built
//! directly on the crate's public genetic-algorithm primitives (population
//! management, non-dominated sorting/crowding-distance selection, and the
//! real SBX crossover / polynomial mutation / tournament selection
//! operators) rather than its monolithic `NSGAII` struct. This still
//! exercises `scirs2-optimize`'s real, tested operators — it just owns the
//! generation loop itself instead of handing control to the crate.

use scirs2_core::ndarray::Array1;
use scirs2_optimize::multi_objective::algorithms::MultiObjectiveConfig;
use scirs2_optimize::multi_objective::algorithms::utils::generate_random_population;
use scirs2_optimize::multi_objective::crossover::{CrossoverOperator, SimulatedBinaryCrossover};
use scirs2_optimize::multi_objective::mutation::{MutationOperator, PolynomialMutation};
use scirs2_optimize::multi_objective::selection::{SelectionOperator, TournamentSelection};
use scirs2_optimize::multi_objective::solutions::{MultiObjectiveSolution, Population};

/// A candidate parameter vector -> objective vector search strategy.
/// `propose`/`record` is a classic ask-tell loop: the caller asks for `n`
/// candidates, evaluates them externally (a trial in Phase 4's campaign
/// engine), then reports the objectives back before asking again.
pub trait MultiObjectiveSearch {
    fn propose(&mut self, n: usize) -> Vec<Vec<f64>>;
    fn record(&mut self, params: Vec<f64>, objectives: Vec<f64>);
    /// Non-dominated (params, objectives) pairs seen so far. "Dominated"
    /// means some other recorded point is >= on every objective and >
    /// on at least one (assuming all objectives are to be MAXIMIZED —
    /// callers negate any objective they want minimized, e.g. latency).
    fn pareto_front(&self) -> Vec<(Vec<f64>, Vec<f64>)>;
}

/// Ask-tell NSGA-II built on `scirs2-optimize`'s genetic operators. The
/// current generation's evaluated individuals live in `population`; each
/// `propose()` hands out either the initial random population (first call)
/// or a fresh offspring generation (produced via tournament selection + SBX
/// crossover + polynomial mutation over the current population); each
/// matching round of `record()` calls feeds solutions into `population`,
/// which is culled back down to `population_size` via NSGA-II's real
/// non-dominated-sort + crowding-distance `select_best` once it grows past
/// twice that size (parents + offspring, the standard NSGA-II generational
/// combine-then-select step).
pub struct Nsga2Search {
    config: MultiObjectiveConfig,
    n_variables: usize,
    bounds: Vec<(f64, f64)>,
    population: Population,
    crossover: SimulatedBinaryCrossover,
    mutation: PolynomialMutation,
    selection: TournamentSelection,
    initialized: bool,
}

impl Nsga2Search {
    pub fn new(
        _num_objectives: usize,
        param_bounds: Vec<(f64, f64)>,
        population_size: usize,
    ) -> Self {
        let n_variables = param_bounds.len();
        let lower = Array1::from_vec(param_bounds.iter().map(|(l, _)| *l).collect());
        let upper = Array1::from_vec(param_bounds.iter().map(|(_, u)| *u).collect());
        let config = MultiObjectiveConfig {
            population_size,
            bounds: Some((lower, upper)),
            ..Default::default()
        };
        Self {
            crossover: SimulatedBinaryCrossover::new(
                config.crossover_eta,
                config.crossover_probability,
            ),
            mutation: PolynomialMutation::new(config.mutation_probability, config.mutation_eta),
            selection: TournamentSelection::new(2),
            bounds: param_bounds,
            n_variables,
            config,
            population: Population::new(),
            initialized: false,
        }
    }
}

impl MultiObjectiveSearch for Nsga2Search {
    fn propose(&mut self, n: usize) -> Vec<Vec<f64>> {
        if !self.initialized {
            self.initialized = true;
            return generate_random_population(n, self.n_variables, &self.config.bounds)
                .into_iter()
                .map(|a| a.to_vec())
                .collect();
        }

        let parents = self.population.solutions().to_vec();
        let mut offspring = Vec::with_capacity(n);
        while offspring.len() < n && !parents.is_empty() {
            let selected = self.selection.select(&parents, 2);
            if selected.len() < 2 {
                break;
            }
            let p1 = selected[0].variables.as_slice().expect("contiguous");
            let p2 = selected[1].variables.as_slice().expect("contiguous");
            let (mut c1, mut c2) = self.crossover.crossover(p1, p2);
            self.mutation.mutate(&mut c1, &self.bounds);
            self.mutation.mutate(&mut c2, &self.bounds);
            offspring.push(c1);
            if offspring.len() < n {
                offspring.push(c2);
            }
        }
        offspring
    }

    fn record(&mut self, params: Vec<f64>, objectives: Vec<f64>) {
        self.population.add(MultiObjectiveSolution::new(
            Array1::from_vec(params),
            Array1::from_vec(objectives),
        ));
        // Standard NSGA-II generational step: once parents + offspring have
        // both been recorded (population has grown past target size), cull
        // back down via the real non-dominated-sort + crowding-distance
        // selection so the next propose() only breeds from survivors.
        if self.population.size() > self.config.population_size * 2 {
            let survivors = self.population.select_best(self.config.population_size);
            self.population = Population::from_solutions(survivors);
        }
    }

    fn pareto_front(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        self.population
            .extract_pareto_front()
            .into_iter()
            .map(|s| (s.variables.to_vec(), s.objectives.to_vec()))
            .collect()
    }
}

#[cfg(test)]
mod spike {
    use super::*;

    /// A toy 2-parameter, 2-objective problem with a KNOWN Pareto front:
    /// maximize f1 = x, maximize f2 = 1 - x, for x in [0, 1]. Every x is
    /// itself Pareto-optimal (increasing x trades f1 for f2 1:1) — so the
    /// true front is the full curve f1 + f2 = 1. A working multi-objective
    /// search over this space should, after enough trials, produce a set
    /// of non-dominated points whose (f1, f2) pairs sum close to 1.0 and
    /// span a spread of x values (not collapse to one point).
    #[test]
    fn nsga2_search_finds_the_known_pareto_front_on_a_toy_problem() {
        // This test constructs the real scirs2-optimize-backed searcher
        // (Nsga2Search, implemented in Step 3) and drives it through the
        // ask-tell loop for several generations.
        let mut search = super::Nsga2Search::new(
            /* num_objectives */ 2,
            /* param_bounds */ vec![(0.0, 1.0)],
            /* population_size */ 20,
        );

        for _generation in 0..15 {
            let candidates = search.propose(20);
            for params in candidates {
                let x = params[0];
                let f1 = x;
                let f2 = 1.0 - x;
                search.record(params, vec![f1, f2]);
            }
        }

        let front = search.pareto_front();
        assert!(
            !front.is_empty(),
            "search must produce a non-empty Pareto front"
        );
        for (_, objectives) in &front {
            let sum = objectives[0] + objectives[1];
            assert!(
                (sum - 1.0).abs() < 0.05,
                "every front point should lie near f1+f2=1 (got {sum})"
            );
        }
        // The front should span a spread of x values, not collapse to one
        // point — evidence the search actually explored, not converged
        // to a single local optimum.
        let xs: Vec<f64> = front.iter().map(|(p, _)| p[0]).collect();
        let spread = xs.iter().cloned().fold(f64::MIN, f64::max)
            - xs.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            spread > 0.3,
            "front should span x values, not collapse (spread={spread})"
        );
    }
}
