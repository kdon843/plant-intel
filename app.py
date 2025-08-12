# ------------- Data status (no upload prompt) -------------
st.subheader("Data status")
c1, c2 = st.columns(2)

with c1:
    st.write("**Expected paths**")
    st.code(f"LENIENT_PARQ : {E_LENIENT}")
    st.code(f"PASS_PARQ    : {E_PASSAGES}")
    st.code(f"DETAILS_PARQ : {E_DETAILS}")

with c2:
    st.write("**File exists?**")
    st.write(f"Lenient : {'✅' if os.path.exists(E_LENIENT) else '❌'}")
    st.write(f"Passages: {'✅' if os.path.exists(E_PASSAGES) else '❌'}")
    st.write(f"Details : {'✅' if os.path.exists(E_DETAILS) else '⚠️ (optional)'}")

# Optional: one-click copy from C: to local path
def _safe_copy(src, dst):
    try:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            import shutil
            shutil.copy2(src, dst)
            return True, f"Copied {src} → {dst}"
        return False, f"Source does not exist: {src}"
    except Exception as e:
        return False, f"Copy failed: {e}"

if E_LENIENT.startswith(("C:", "c:")):
    if os.path.exists(E_LENIENT) and not os.path.exists(LOCAL_LENIENT):
        if st.button("Copy lenient dataset into project folder"):
            ok, msg = _safe_copy(E_LENIENT, LOCAL_LENIENT)
            (st.success if ok else st.error)(msg)
            st.info("Click 'Rerun' to reload once the file is copied.")

# ------------- Stop if no dataset is found -------------
ready = os.path.exists(E_LENIENT) or os.path.exists(E_PASSAGES) or os.path.exists(E_DETAILS)
if not ready:
    st.warning(
        "No recommender data found yet. "
        "Make sure the paths above exist on this machine."
    )
    st.stop()
