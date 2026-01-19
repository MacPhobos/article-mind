<script lang="ts">
	// Root layout - wraps all pages
	let { children } = $props();
</script>

<main>
	{@render children()}
</main>

<style>
	:global(body) {
		margin: 0;
		padding: 0;
		font-family:
			system-ui,
			-apple-system,
			sans-serif;
	}

	main {
		min-height: 100vh;
	}
</style>
