/* State shared between the 3D page and the dashboard page (same origin, localStorage).
   The 3D page owns the bin state and the camera snapshot; the dashboard reads them,
   and can also ask for the bin to be emptied or overflowed. */
(function () {
	var K_FULL = "binwatch.full",
		K_SNAP = "binwatch.snap";
	function get(k) {
		try {
			return localStorage.getItem(k);
		} catch (e) {
			return null;
		}
	}
	function set(k, v) {
		try {
			localStorage.setItem(k, v);
		} catch (e) {}
	}
	window.binStore = {
		getFull: function () {
			var v = get(K_FULL);
			return v === null ? true : v === "1";
		},
		setFull: function (f) {
			set(K_FULL, f ? "1" : "0");
		},
		getSnap: function () {
			try {
				return JSON.parse(get(K_SNAP) || "null");
			} catch (e) {
				return null;
			}
		},
		setSnap: function (s) {
			set(K_SNAP, JSON.stringify(s));
		},
		/* fires in the OTHER open page when it changes something */
		onChange: function (fn) {
			window.addEventListener("storage", function (e) {
				if (e.key === K_FULL) fn({ full: window.binStore.getFull() });
				else if (e.key === K_SNAP) fn({ snap: window.binStore.getSnap() });
			});
		},
	};
})();
