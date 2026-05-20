//! A [`Tso`] wrapper that allows the inner impl to be hot-swapped.
//!
//! Production deployments start with [`crate::LocalTso`] until the
//! metadata Raft is up, then atomically swap to a `MetaLeasedTso`
//! (in `veda-coordinator`). Callers hold a single
//! `Arc<SwappableTso>` and never need to re-read the field — the
//! `.now()` and `.allocate()` calls always hit the current inner.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::{Ts, TsRange, Tso, TsoError};

#[derive(Debug)]
pub struct SwappableTso {
    inner: RwLock<Arc<dyn Tso>>,
}

impl SwappableTso {
    /// Build a swappable TSO around an initial inner impl.
    pub fn new(initial: Arc<dyn Tso>) -> Self {
        Self {
            inner: RwLock::new(initial),
        }
    }

    /// Replace the inner impl. Subsequent `now` / `allocate`
    /// calls go through the new impl. Atomic.
    pub async fn swap(&self, new: Arc<dyn Tso>) {
        *self.inner.write().await = new;
    }

    /// Borrow a clone of the current inner — useful for code
    /// that needs to call multiple methods atomically without
    /// re-locking.
    pub async fn current(&self) -> Arc<dyn Tso> {
        self.inner.read().await.clone()
    }
}

#[async_trait]
impl Tso for SwappableTso {
    async fn allocate(&self, n: u32) -> Result<TsRange, TsoError> {
        let inner = self.inner.read().await.clone();
        inner.allocate(n).await
    }

    async fn now(&self) -> Result<Ts, TsoError> {
        let inner = self.inner.read().await.clone();
        inner.now().await
    }
}
