<script lang="ts">
	let appName = $state('Article Mind');
</script>

<div class="container">
	<h1>Welcome to {appName}</h1>
	<p>SvelteKit frontend is running!</p>
</div>

<style>
	.container {
		max-width: 800px;
		margin: 0 auto;
		padding: 2rem;
		text-align: center;
	}

	h1 {
		color: #333;
		margin-bottom: 1rem;
	}
</style>
