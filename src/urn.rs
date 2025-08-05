use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

use crate::util::hypergeometric_sample;

// TODO: I think almost every single instance of "State" and "usize" in this file are being used backwards :)
type State = usize;

/// Samples a discrete uniform random number in the range [low, high].
/// Note inclusive on both ends
pub fn sample_discrete_uniform(rng: &mut SmallRng, low: usize, high: usize) -> usize {
    // <DiscreteUniform as rand::distributions::Distribution<i64>>::sample(&discrete_uniform, rng)
    //     as usize
    rng.random_range(low..=high)
}

/// Data structure for a multiset that supports fast random sampling.
pub struct Urn {
    pub config: Vec<usize>,
    pub order: Vec<usize>,
    pub size: usize,
    rng: SmallRng,
}

impl Urn {
    /// Create a new Urn object.
    pub fn new(config: Vec<State>, seed: Option<u64>) -> Self {
        let size = config.iter().sum();
        let order = (0..config.len()).collect();
        let rng = if let Some(s) = seed {
            SmallRng::seed_from_u64(s)
        } else {
            SmallRng::from_os_rng()
        };

        let mut urn = Urn {
            config,
            order,
            size,
            rng,
        };

        urn.sort();

        urn
    }

    /// Updates self.order.
    ///
    /// Uses insertion sort to maintain that
    ///   config[order[0]] >= config[order[1]] >= ... >= config[order[q]].
    /// This method is used to have O(q) time when order is almost correct.
    pub fn sort(&mut self) {
        for i in 1..self.config.len() {
            // See if the entry at order[i] needs to be moved earlier.
            // Recursively, we have ensured that order[0], ..., order[i-1] have the correct order.
            let o_i = self.order[i];
            // j will be the index where order[i] should be inserted to.
            let mut j = i;
            while j > 0 && self.config[o_i] > self.config[self.order[j - 1]] {
                j -= 1;
            }

            // Index at order[i] will get moved to order[j], and all indices order[j], ..., order[i-1] get right shifted
            // First do the right shift, moving order[i-k] for k = 1, ..., i-j
            for k in 1..(i - j + 1) {
                self.order[i + 1 - k] = self.order[i - k];
            }
            self.order[j] = o_i;
        }
    }

    /// Samples and removes one element, returning its index.
    pub fn sample_one(&mut self) -> Result<State, String> {
        if self.size <= 0 {
            return Err("Cannot sample from empty urn".to_string());
        }

        // Generate random integer in [0, self.size-1]
        let x = sample_discrete_uniform(&mut self.rng, 0, self.size - 1);

        let mut i = 0;
        let mut x: i64 = x as i64;
        let mut index = 0;

        while x >= 0 {
            index = self.order[i];
            x -= self.config[index] as i64;
            i += 1;
        }

        // Decrement the count for the sampled element
        self.config[index] -= 1;
        self.size -= 1;

        Ok(index)
    }

    /// Adds one element at index.
    pub fn add_to_entry(&mut self, index: usize, amount: i64) {
        assert!(self.config[index] as i64 + amount >= 0);
        self.config[index] = (self.config[index] as i64 + amount) as State;
        assert!(self.size as i64 + amount >= 0);
        self.size = (self.size as i64 + amount) as usize;
    }

    /// Samples n elements, writing them into the vector v.
    ///
    /// This method is implemented only to make testing easier, but sample_vector_impl
    /// should be called within Rust code, since it does not allocate new memory as this does.
    ///
    /// Args:
    ///     n: number of elements to sample
    ///     v: the array to write the output vector in
    ///         (this is faster than re-initializing an output array)
    ///         
    /// Returns:
    ///     idx_nx: the index (into self.order) of the last nonzero entry
    ///         v[self.order[i]] for i in range(nz) can then skip looping over
    ///         any entries after the last nonzero entry.
    pub fn sample_vector(&mut self, n: usize, v: &mut [State]) -> Result<usize, String> {
        let mut n = n;
        let mut i: usize = 0;
        let mut total: usize = self.size;
        for j in 0..v.len() {
            v[j] = 0;
        }

        while n > 0 && i < self.config.len() - 1 {
            let index = self.order[i];
            let successes = self.config[index];
            let h = hypergeometric_sample(total, successes, n, &mut self.rng);
            total -= self.config[index];

            v[index] = h;
            assert!(n as i64 - h as i64 >= 0);
            n -= h;
            assert!(self.size as i64 - h as i64 >= 0);
            self.size -= h;
            assert!(self.config[index] as i64 - h as i64 >= 0);
            self.config[index] -= h;
            i += 1;
        }

        if n != 0 {
            assert!(n > 0);
            v[self.order[i]] = n;
            assert!(self.config[self.order[i]] as i64 - n as i64 >= 0);
            self.config[self.order[i]] -= n;
            assert!(self.size as i64 - n as i64 >= 0);
            self.size -= n;
            i += 1;
        }

        Ok(i)
    }

    /// Adds a vector of elements to the urn.
    pub fn add_vector(&mut self, vector: &Vec<State>) {
        for i in 0..self.config.len() {
            let count = vector[i];
            self.config[i] += count;
            self.size += count;
        }
    }

    /// Set the counts back to zero.
    pub fn reset(&mut self) {
        for i in 0..self.config.len() {
            self.config[i] = 0;
        }
        self.size = 0;
    }

    /// This mimics creating a new Urn, but instead we replace
    /// the current config with the new one. This is useful because
    /// we can save the RNG. Otherwise we would have to create a
    /// new Urn with the same RNG and this avoid borrowship issues.
    pub fn reset_config(&mut self, config: &Vec<State>) {
        for i in 0..self.config.len() {
            self.config[i] = config[i];
        }
        self.size = config.iter().sum();
        self.order = (0..config.len()).collect();
        self.sort();
    }
}
